import gradio as gr
import paddle, os
from ppdiffusers import DiffusionPipeline

stable_model_list = [
    "paddle/BraV5", "paddle/dark-sushi-25d",
]

supported_scheduler = [
    "pndm",
    "lms",
    "euler",
    "euler-ancestral",
    "dpm-multi",
    "dpm-single",
    "unipc-multi",
    "ddim",
    "ddpm",
    "deis-multi",
    "heun",
    "kdpm2-ancestral",
    "kdpm2",
]

class StableDiffusionText2ImageGenerator:
    def __init__(self):
        self.pipe = None

    def load_model(
        self,
        model_path,
        scheduler,
    ):
        if self.pipe is None:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_path, from_hf_hub=True, safety_checker=None, paddle_dtype=paddle.float16, custom_pipeline="webui_stable_diffusion.py"
            )
            self.pipe.LORA_DIR=os.path.join(os.getcwd(), "lora")
            self.pipe.TI_DIR=os.path.join(os.getcwd(), "textual_inversion")

        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.switch_scheduler(scheduler)
        return self.pipe

    def generate_image(
        self,
        model_path: str,
        prompt: str,
        negative_prompt: str,
        clip_skip: int,
        scheduler: str,
        guidance_scale: int,
        num_inference_step: int,
        height: int,
        width: int,
        seed_generator=-1,
    ):
        pipe = self.load_model(
            model_path=model_path,
            scheduler=scheduler,
        )

        if not seed_generator == -1:
            paddle.seed(seed_generator)

        images = pipe(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            clip_skip=clip_skip,
            num_inference_steps=num_inference_step,
            guidance_scale=guidance_scale, 
        ).images

        return images

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    text2image_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Prompt",
                        value="koh_daisyridley <lora:epi_noiseoffset2:1.0>",
                        show_label=False,
                    )
                    text2image_negative_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Negative Prompt",
                        value="(low quality:1.3) (worst quality:1.3) bad_prompt_version2 bad-artist bad-artist-anime bad-hands-5 bad-image-v2-39000 EasyNegative EasyNegativeV2 ng_deepnegative_v1_75t verybadimagenegative_v1.3",
                        show_label=False,
                    )
                    with gr.Row():
                        with gr.Column():
                            text2image_model_path = gr.Dropdown(
                                choices=stable_model_list,
                                value=stable_model_list[0],
                                label="Text-Image Model Id",
                            )
                            text2image_guidance_scale = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=7.5,
                                label="Guidance Scale",
                            )
                            text2image_num_inference_step = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=50,
                                label="Num Inference Step",
                            )
                            text2image_clip_skip = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    step=1,
                                    label="Clip Skip",
                                    value=1,
                            )
                        with gr.Row():
                            with gr.Column():
                                text2image_scheduler = gr.Dropdown(
                                    choices=supported_scheduler,
                                    value=supported_scheduler[3],
                                    label="Scheduler",
                                )
                                text2image_width = gr.Slider(
                                    minimum=128,
                                    maximum=1280,
                                    step=32,
                                    value=512,
                                    label="Image Width",
                                )
                                text2image_height = gr.Slider(
                                    minimum=128,
                                    maximum=1280,
                                    step=32,
                                    value=512,
                                    label="Image Height",
                                )
                                text2image_seed_generator = gr.Number(
                                    label="Seed(-1 for random)",
                                    value=-1,
                                )

                    text2image_predict = gr.Button(value="Generator")

                with gr.Column():
                    output_image = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=(1, 2), height=200)

            text2image_predict.click(
                fn=StableDiffusionText2ImageGenerator().generate_image,
                inputs=[
                    text2image_model_path,
                    text2image_prompt,
                    text2image_negative_prompt,
                    text2image_clip_skip,
                    text2image_scheduler,
                    text2image_guidance_scale,
                    text2image_num_inference_step,
                    text2image_height,
                    text2image_width,
                    text2image_seed_generator,
                ],
                outputs=output_image,
            )

camenduru = """
    🐣 Please follow me for new updates [https://github.com/camenduru](https://github.com/camenduru) <br />
    #### Tutorial
    - Please add your lora model in `work/lora` folder if your lora file name `epi_noiseoffset2.safetensors` your trigger token is `<lora:epi_noiseoffset2:1.0>`
    - Please add your textual inversion model in `work/textual_inversion` if your textual inversion file name `koh_daisyridley.pt` your trigger token is `koh_daisyridley`
    #### Installed positive LoRAs
    - \<lora:epi_noiseoffset2:1.0\>
    - \<lora:Japanese-doll-likeness:1.0\>
    - \<lora:Korean-doll-likeness:1.0\>
    - \<lora:Taiwan-doll-likeness:1.0\>
    #### Installed positive embeds
    - koh_daisyridley
    #### Installed negative embeds
    - verybadimagenegative_v1.3
    - bad_prompt_version2
    - ng_deepnegative_v1_75t
    - bad-artist-anime
    - bad-image-v2-39000
    - bad-hands-5
    - bad-artist
    """

def diffusion_app():
    app = gr.Blocks()
    with app:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Text2Img"):
                    StableDiffusionText2ImageGenerator.app()
        gr.Markdown(camenduru)
    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)

if __name__ == "__main__":
    diffusion_app()
