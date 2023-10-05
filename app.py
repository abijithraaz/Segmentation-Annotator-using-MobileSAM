import os
import gradio as gr
import numpy as np
from PIL import Image
import argparse
import pathlib
from torch.nn import functional as F

from show import *
from per_segment_anything import sam_model_registry, SamPredictor


parser = argparse.ArgumentParser()
parser.add_argument("-op", "--output-path", type=str, default='default')
args = parser.parse_args()
   

def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
        
    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    
    return topk_xy, topk_label, last_xy, last_label

def reset_data():
    global cache_data
    cache_data = None

def inference_scribble(image):
    # in context image and mask
    ic_image = image["image"]
    ic_mask = image["mask"]
    ic_image = np.array(ic_image.convert("RGB"))
    ic_mask = np.array(ic_mask.convert("RGB"))
    
    # sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth' # SAM Model
    sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt' # MobileSAM
    # sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda() #SAM loading
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt) #SAM loading
    # sam = sam_model_registry[sam_type](checkpoint=sam_ckpt) # MObileSAM loading
    predictor = SamPredictor(sam)
    
    # Image features encoding
    ref_mask = predictor.set_image(ic_image, ic_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]
    
    # Target feature extraction
    print("======> Obtain Location Prior" )
    target_feat = ref_feat[ref_mask > 0]
    target_embedding = target_feat.mean(0).unsqueeze(0)
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    target_embedding = target_embedding.unsqueeze(0)
    
    test_image = ic_image
    outputs = []

    print("======> Testing Image")   
    # Image feature encoding
    predictor.set_image(test_image)
    test_feat = predictor.features.squeeze()

    # Cosine similarity
    C, h, w = test_feat.shape
    test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
    test_feat = test_feat.reshape(C, h * w)
    sim = target_feat @ test_feat

    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    sim = predictor.model.postprocess_masks(
        sim,
        input_size=predictor.input_size,
        original_size=predictor.original_size).squeeze()
    
    # Positive-negative location prior
    topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
    topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
    topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

    # Obtain the target guidance for cross-attention layers
    sim = (sim - sim.mean()) / torch.std(sim)
    sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
    attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

    # First-step prediction
    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy, 
        point_labels=topk_label, 
        multimask_output=True,
        attn_sim=attn_sim,  # Target-guided Attention
        target_embedding=target_embedding  # Target-semantic Prompting
    )
    best_idx = 0

    # Cascaded Post-refinement-1
    masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                mask_input=logits[best_idx: best_idx + 1, :, :], 
                multimask_output=True)
    best_idx = np.argmax(scores)

    # Cascaded Post-refinement-2
    y, x = np.nonzero(masks[best_idx])
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    input_box = np.array([x_min, y_min, x_max, y_max])
    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        box=input_box[None, :],
        mask_input=logits[best_idx: best_idx + 1, :, :], 
        multimask_output=True)
    best_idx = np.argmax(scores)

    final_mask = masks[best_idx]
    mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
    mask_colors[final_mask, :] = np.array([[128, 0, 0]])
    # Save annotations

    return [Image.fromarray((mask_colors * 0.6 + test_image * 0.4).astype('uint8'), 'RGB'), 
            Image.fromarray((mask_colors ).astype('uint8'), 'RGB')]


with gr.Blocks() as demo:
    gr.Markdown("# Segmentation-Annotator-using-MobileSAM")
    gr.Markdown("To start, input an image, then use the brush to create dots on the object which you want to segment, don't worry if your dots aren't perfect as the code will find the middle of each drawn item. Then press the segment button to create masks for the object that the dots are on.")
    gr.Markdown("## Demo to run Mobile Segment Anything base model")
    gr.Markdown("""This app uses the [MobileSAM](https://github.com/ChaoningZhang/MobileSAM.git) model to get a mask from a points in an image.""")
    gr.Markdown("""Full code can be found here [SourceCode](https://github.com/ChaoningZhang/MobileSAM.git)""")
    with gr.Row():
        image_input = gr.Image(label="[Stroke] Draw on Image", tool='sketch',type='pil')
        image_output1 = gr.Image(type="pil", label="Mask with Image")
    with gr.Row():
        # examples = gr.Examples(examples=["./cardamage_example/0006.JPEG",
        #                                 "./cardamage_example/0008.JPEG",
        #                                 "./cardamage_example/0206.JPEG"],
        #                       inputs=image_input)
        examples = gr.Examples(examples="./cardamage_example", inputs=image_input)
        image_output2 = gr.Image(type="pil", label="Mask")
    
    image_button = gr.Button("Genarate-Segment-Mask", variant='primary')

    image_button.click(inference_scribble, inputs=image_input, outputs=[image_output1, image_output2])
    image_input.upload(reset_data)

demo.launch()
