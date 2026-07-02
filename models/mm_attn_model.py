import torch
from .model import Model
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch.nn.functional as F
from .utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def add_diffusion_noise(image_tensor, noise_step): # from VCD
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd

class MultimodalAttentionModel(Model):
    def __init__(self, instruction, image, config):
        super().__init__(config)
        self.name = config["model_info"]["name"]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        model_id = config["model_info"]["model_id"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llama3-llava-next-8b-hf",
            torch_dtype=torch.float16,
            device_map=None,
        ).to(self.device).eval()

        self.top_k = 50
        self.top_p = None

        if config["params"]["important_heads"] == "all":
            attn_size = self.get_map_dim(instruction, image)
            self.important_heads = [[i, j] for i in range(
                attn_size[0]) for j in range(attn_size[1])]
        else:
            self.important_heads = config["params"]["important_heads"]


    def get_map_dim(self, instruction, image):
        _, _, attention_maps, _, _, _ = self.inference(instruction, image)
        attention_map = attention_maps[0]
        return len(attention_map), attention_map[0].shape[1]

    def inference(self, instruction, image, max_output_tokens=None, noise_step=None):
        """
        Multimodal inference for LLaVA-NeXT.

        instruction:
            trusted task instruction / system instruction

        img:
            PIL image, already opened by Image.open(...).convert("RGB")
        """

        # conversation = [
        #     {
        #         "role": "system",
        #         "content": instruction,
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image"},
        #         ],
        #     },
        # ]

        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        if noise_step is not None:
            image_tensor = add_diffusion_noise(image_tensor, noise_step)


        model_inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        generated_tokens = []
        generated_probs = []
        attention_maps = []
        input_ids = model_inputs.input_ids.to(self.device)
        attention_mask = model_inputs.attention_mask.to(self.device)

        input_tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids[0])

        if max_output_tokens != None:
            n_tokens = max_output_tokens
        else:
            n_tokens = self.max_output_tokens

        with torch.no_grad():
            for i in range(n_tokens):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    pixel_values=model_inputs["pixel_values"],
                    image_sizes=model_inputs["image_sizes"],
                    return_dict=True,
                )

                # only need to compute ranges once, at the first generation step
                if i == 0:
                    instruction_range = self._get_instruction_range(
                        model_inputs["input_ids"],
                        instruction,
                    )

                    image_range = self._get_image_range(
                        model_inputs,
                        output,
                    )

                    ranges = (instruction_range, image_range)
                    
                logits = output.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                
                # next_token_id = logits.argmax(dim=-1).squeeze()
                next_token_id = sample_token(
                    logits[0], top_k=self.top_k, top_p=self.top_p, temperature=1.0)[0]

                generated_probs.append(probs[0, next_token_id.item()].item())
                generated_tokens.append(next_token_id.item())

                attention_map = [
                    attention.detach().cpu().half()
                    for attention in output['attentions']
                ]
                attention_map = [
                    torch.nan_to_num(attention, nan=0.0) 
                    for attention in attention_map
                ]
                attention_map = get_last_attn(attention_map)
                attention_maps.append(attention_map)

                if next_token_id.item() == self.processor.tokenizer.eos_token_id:
                    break

                input_ids = torch.cat(
                    [input_ids, next_token_id.view(1, 1).to(input_ids.device)],
                    dim=-1,
                )

                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (1, 1),
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        ),
                    ],
                    dim=-1,
                )

        output_tokens = [self.processor.tokenizer.decode(
            token, skip_special_tokens=True) for token in generated_tokens]
        generated_text = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True)

        return generated_text, output_tokens, attention_maps, input_tokens, ranges, generated_probs
    
    def _find_subsequence(self, sequence, pattern):
        """
        Find the first occurrence of pattern in sequence.
        Return (start, end) if found, otherwise None.
        """
        pattern_len = len(pattern)

        if pattern_len == 0:
            return None

        for start in range(len(sequence) - pattern_len + 1):
            if sequence[start:start + pattern_len] == pattern:
                return (start, start + pattern_len)

        return None


    def _get_instruction_range(self, input_ids, instruction):
        """
        Find instruction token range in the original text input_ids.
        This is before image-token expansion.
        """
        full_ids = input_ids[0].tolist()

        instruction_ids = self.processor.tokenizer(
            instruction,
            add_special_tokens=False
        )["input_ids"]

        instruction_range = self._find_subsequence(full_ids, instruction_ids)

        # print("instruction:", repr(instruction), flush=True)
        # print("instruction_ids:", instruction_ids, flush=True)
        # print("instruction_tokens:", self.processor.tokenizer.convert_ids_to_tokens(instruction_ids), flush=True)

        # print("full_ids first 30:", full_ids[:30], flush=True)
        # print("full_tokens first 30:", self.processor.tokenizer.convert_ids_to_tokens(full_ids[:30]), flush=True)

        return instruction_range


    def _get_image_range(self, model_inputs, outputs):
        """
        Get image token block range for LLaVA-NeXT.

        In LLaVA-NeXT, a single image is expanded by the processor into many
        <image> placeholder tokens, so we should not expect exactly one image token.
        """
        input_ids = model_inputs["input_ids"]

        image_token_id = self.model.config.image_token_index
        image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]

        if len(image_positions) == 0:
            raise ValueError("No <image> token found in input_ids.")

        # For single-image LLaVA-NeXT input, these positions should form one contiguous block.
        image_start = image_positions[0].item()
        image_end = image_positions[-1].item() + 1

        # Optional sanity check: image tokens should be contiguous
        expected = torch.arange(
            image_start,
            image_end,
            device=image_positions.device,
            dtype=image_positions.dtype,
        )

        if not torch.equal(image_positions, expected):
            raise ValueError(
                "Image token positions are not contiguous. "
                "This may indicate multi-image input or an unexpected chat template."
            )

        merged_seq_len = outputs.attentions[0].shape[-1]

        assert image_start >= 0
        assert image_end <= merged_seq_len

        return (image_start, image_end)
