import matplotlib.pyplot as plt
import torch

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size, task):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)
        
        # Move model to same device as input if not already
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        print("\nInput tensor shape:", input_tensor.shape)
        
        # Set model to eval mode for visualization
        self.model.eval()
        attn_maps = []
        with torch.no_grad():
            if task == "encoder":
                _, attn_maps = self.model(input_tensor)
            # if task == "part2":
            else:
                _, _, attn_maps, probs = self.model(input_tensor)
                
        print("\nNumber of layers with attention maps:", len(attn_maps))
        
        for layer_idx, layer_attn_maps in enumerate(attn_maps):
            print(f"Processing Layer {layer_idx + 1}")
            print(f"Number of attention heads in layer: {len(layer_attn_maps)}")
            
            for head_idx, head_attn_map in enumerate(layer_attn_maps):
                att_map = head_attn_map.squeeze(0).cpu().numpy()
                
                # Verification step
                row_sums = head_attn_map[0].sum(dim=-1).cpu()
                if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
                    print(f"Warning: Layer {layer_idx+1}, Head {head_idx+1} - Row sums:", row_sums)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                cax = ax.imshow(att_map, cmap='hot', interpolation='nearest')
                ax.set_title(f"Layer {layer_idx + 1}, Head {head_idx + 1}")
                plt.colorbar(cax)
                plt.savefig(f"{task}_attention_map_L{layer_idx+1}_H{head_idx+1}.png")
                # plt.show()
                # plt.close()
        return