# flexible model
# offline editing
# online inference
import logging, os, time
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor
from openai import OpenAI

from LAMB.lamb.lamb_main import LAMB
from EMCID.emcid.emcid_main import apply_emcid_to_text_encoder_debias
from EMCID.emcid.emcid_hparams import EMCIDHyperParams
from EMCID.experiments.emcid_test import set_weights

from EMCID.scripts.refact_benchmark_eval import set_seed

class BATIP:
    def __init__(self, 
            global_seed=2025,
            device="cuda:0",
            log_dir="./results/logs",
            image_dir="./results/images",
            openai_client=None,
            prompt_dir="./LAMB/prompts",
            mem_dir="./LAMB/memories",
            hparam_dir="./EMCID/hparams",
            cache_dir="./EMCID/cache",
            ):
        logging.info("[BATIP] Initializing...")
        self.global_seed = global_seed
        self.device = device
        
        self.log_dir = log_dir
        self.image_dir = image_dir

        self.openai_client = openai_client
        self.prompt_dir = prompt_dir
        self.mem_dir = mem_dir
        self.hparam_dir = hparam_dir
        self.cache_dir = cache_dir

        self.lamb = LAMB(self.openai_client, self.prompt_dir, self.log_dir, self.mem_dir)
        self.pipe = None
        self.hparams = None
        self.cache_name = None

        self.user_input = None
        self.source2dest = None
        self.requests = None
        logging.info("[BATIP] Successfully initialized.")

    def generate_labels(self, 
            model="gpt-4o-2024-05-13", 
            apply_rag=False, 
            apply_reflection=False, 
            reflect_iter=1
            ):
        logging.info("[LAMB] Generating labels with user input...")
        logging.info("[LAMB] ChatGPT model: %s", model)
        logging.info("[LAMB] Apply RAG: %s", apply_rag)
        logging.info("[LAMB] Apply reflection: %s", apply_reflection)
        logging.info("[LAMB] Reflection iteration: %d", reflect_iter)
        total_prompt_tokens = 0
        total_completion_tokens = 0
        self.user_input =  self.lamb.get_user_input()
        logging.info(f"[LAMB] User input: \"{self.user_input}\"")
        messages = self.lamb.integrate_prompts(
            user_input=self.user_input, 
            apply_rag=apply_rag, 
            apply_reflection=apply_reflection)

        response, prompt_tokens, completion_token = self.lamb.predict_bias(model=model, messages=messages)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_token

        if apply_reflection:
            for iter in range(reflect_iter):
                messages = self.lamb.integrate_prompts(
                    user_input=self.user_input, 
                    prev_response=response, 
                    apply_reflection=apply_reflection)
                response = self.lamb.predict_bias(model=model, messages=messages)
        
        _, bias, self.source2dest = self.lamb.extract_attributes(response=response)
        logging.info(f"[LAMB] Potential bias in adjectives: {bias}")

        logging.info(f"[LAMB] Total Prompt Tokens: {total_prompt_tokens}, Total Completion Tokens: {total_completion_tokens}")
        logging.info("[LAMB] Successfully generated labels.")

    def prepare_requests(self,
            template=["a photo of {}",
                    "an image of {}",
                    "{}"],
            ):
        logging.info("[LAMB] Preparing requests...")
        logging.info(f"[LAMB] Template: {template}")

        self.requests = []
        for entity in self.source2dest:
            request = {
                "prompts": template, 
                "seed": self.global_seed,
                "source": entity,
                "dests": self.source2dest[entity],
            }
            self.requests.append(request)
        # # for testing
        # self.requests = [{
        #         "prompts": template, 
        #         "seed": self.global_seed,
        #         "source": "a farmer",
        #         "dests": ["a male farmer", "a female farmer"],
        #     }]
    
    def init_pipe(self,
            diffusion_model="CompVis/stable-diffusion-v1-4",
            hparam_name="dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01",
            ):
        logging.info("[EMCID] Initializing Stable Diffusion pipeline...")
        logging.info(f"[EMCID] Diffusion model: {diffusion_model}")
        logging.info(f"[EMCID] Hparam name: {hparam_name}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=diffusion_model,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        self.hparams = EMCIDHyperParams.from_json(f"{self.hparam_dir}/{hparam_name}.json")
        self.cache_name = f"{self.cache_dir}/{hparam_name}/debiasing/"

        self.hparams = set_weights(self.hparams, mom2_weight, edit_weight)
        logging.info(f"[EMCID] Successfully initialized Stable Diffusion pipeline.")
    
    def edit_concept(self,
            edit_iter=20,
            mom2_weight=None,
            edit_weight=None,
            return_orig_text_model=True,
            recompute_factors=False,
            verbose=False,
            ):
        logging.info("[EMCID] Editing concept...")
        logging.info(f"[EMCID] Edit iteration: {edit_iter}")
        logging.info(f"[EMCID] Mom2 weight: {mom2_weight}")
        logging.info(f"[EMCID] Edit weight: {edit_weight}")
        logging.info(f"[EMCID] Return original text model: {return_orig_text_model}")
        logging.info(f"[EMCID] Recompute factors: {recompute_factors}")
        logging.info(f"[EMCID] Verbose: {verbose}")
        logging.info(f"[EMCID] Cache name: {self.cache_name}")
        self.pipe, orig_text_encoder = apply_emcid_to_text_encoder_debias(
            pipe=self.pipe,
            requests=self.requests,
            hparams=self.hparams,
            device=self.device,
            mom2_weight=mom2_weight,
            edit_weight=edit_weight,
            cache_name=self.cache_name,
            max_iter=edit_iter,
            return_orig_text_model=return_orig_text_model,
            recompute_factors=recompute_factors,
            verbose=verbose,
            )
        self.pipe.text_encoder = orig_text_encoder
        logging.info("[EMCID] Successfully edited concept.")

    def generate_imgs(self,
            imgs_per_prompt: int=24,
            after_edit: bool=False,
        ):
        logging.info("[EvaBiM] Generating images for debias evaluation...")
        logging.info(f"[EvaBiM] Images per prompt: {imgs_per_prompt}")
        logging.info(f"[EvaBiM] After edit: {after_edit}")
        logging.info(f"[EvaBiM] Image dir: {self.image_dir}")
        with torch.no_grad():
            for request in self.requests:
                source = request["source"]
                print("generating for seed: ", self.global_seed)
                print("generating for source: ", source)
                set_seed(self.global_seed)
                save_dir = f"{self.image_dir}/{source}/after/seed{self.global_seed}" if after_edit else f"{self.image_dir}/{source}/before/seed{self.global_seed}"
                if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                for i in range(imgs_per_prompt):
                    if os.path.exists(f"{save_dir}/idx_{i}.png"):
                        continue
                    img = self.pipe([source], guidance_scale=7.5).images[0]
                    img.save(f"{save_dir}/idx_{i}.png")
                    logging.info("[EvaBiM] Successfully generated image %d for seed %d.", i, self.global_seed)

    def eval_ratio(self,
            after_edit: bool=False,
            ):
        logging.info("[EvaBiM] Evaluating ratio...")
        logging.info(f"[EvaBiM] After edit: {after_edit}")
        logging.info(f"[EvaBiM] Image dir: {self.image_dir}")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device) 
        clip_model.eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        for request in self.requests:
            source = request["source"]
            dests = request["dests"]
            cnts = [0] * len(dests)

            imgs_dir = f"{self.image_dir}/{source}/after/seed{self.global_seed}" if after_edit else f"{self.image_dir}/{source}/before/seed{self.global_seed}"
            for img_name in os.listdir(imgs_dir):
                img = Image.open(f"{imgs_dir}/{img_name}")
                try:
                    inputs = processor(
                                text=dests, 
                                images=[img], 
                                return_tensors="pt", 
                                padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = clip_model(**inputs)

                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=-1)
                    max_idx = probs.argmax(dim=-1).item()
                    cnts[max_idx] += 1
                except OSError:
                    # remove the corrupted image
                    os.remove(f"{imgs_dir}/{img_name}")
                    logging.warning(f"[EvaBiM] Removed corrupted image: {img_name}")
            
            ratio = [cnts[i] / sum(cnts) for i in range(len(dests))]
            logging.info(f"[EvaBiM] Ratio for \"{source}\" to {dests}: {ratio}")
            ratio_std = np.std(ratio)
            baseline = [1 / len(cnts) for _ in range(len(dests))]
            delta = sum([abs(ratio[i]-baseline[i]) / baseline[i] for i in range(len(dests))])
            for i in range(len(ratio)):
                if ratio[i] == 0:
                    ratio[i] = 1e-10  # Avoid log(0) by replacing with a small number
            kl = np.sum([ratio[i] * np.log(ratio[i] / baseline[i]) for i in range(len(dests))])
            logging.info(f"[EvaBiM] Ratio std: {ratio_std}, Ratio delta: {delta}, Ratio KL: {kl}")
            

if __name__ == '__main__':
    # set device
    global_seed = 2025
    device = "cuda:2"

    # log config
    timestamp = time.strftime('%Y%m%d-%H%M%S')

    LOG_DIR = f'./results/{timestamp}/logs'
    IMAGE_DIR = f'./results/{timestamp}/images'

    OPENAI_CLIENT = OpenAI(
        base_url = "",
        api_key = ""
        )

    # LAMB config
    LAMB_DIR = './LAMB'
    PROMPT_DIR = f'{LAMB_DIR}/prompts'
    MEM_DIR = f'{LAMB_DIR}/results/{timestamp}/memories'

    gpt_model = "gpt-4o-2024-05-13"
    apply_rag = False
    apply_reflection = False
    reflect_iter = 1

    template = ["a photo of {}",
                "an image of {}",
                "{}"]

    # EMCID config
    EMCID_DIR = './EMCID'
    HPARAM_DIR = f'{EMCID_DIR}/hparams'
    CACHE_DIR = f'{EMCID_DIR}/results/{timestamp}/cache'
    
    diffusion_model = "CompVis/stable-diffusion-v1-4"
    hparam_name = "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01"

    # fine-tune config
    # edit_iter = 4 # for testing
    # mom2_weight = 4000
    # edit_weight = 0.5
    edit_iter = 1000
    mom2_weight = None
    edit_weight = None
    return_orig_text_model = True
    recompute_factors = False
    verbose = False

    # inference config
    # imgs_per_prompt = 4 # for testing
    imgs_per_prompt = 50

    # eval config
    clip_model = "openai/clip-vit-base-patch16"


    # Create directories if they do not exist
    for dir in [LOG_DIR, IMAGE_DIR, MEM_DIR, CACHE_DIR]:
        os.makedirs(dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        filename=f"{LOG_DIR}/batip.log",
                        filemode='w')

    logging.info("[BATIP] Global seed: %d", global_seed)
    logging.info("[BATIP] Started on device: %s", device)
    logging.info(f"[BATIP] Log dir: {LOG_DIR}")
    logging.info(f"[BATIP] Image dir: {IMAGE_DIR}")

    logging.info(f"[BATIP] ChatGPT model: {gpt_model}")
    logging.info(f"[BATIP] Diffusion model: {diffusion_model}")
    logging.info(f"[BATIP] Hparam name: {hparam_name}")
    logging.info(f"[BATIP] Clip model: {clip_model}")

    batip = BATIP(
        global_seed=global_seed,
        device=device,
        log_dir=LOG_DIR,
        image_dir=IMAGE_DIR,
        openai_client=OPENAI_CLIENT,
        prompt_dir=PROMPT_DIR,
        mem_dir=MEM_DIR,
        hparam_dir=HPARAM_DIR,
        cache_dir=CACHE_DIR,
    )

    batip.generate_labels(
        model=gpt_model, 
        apply_rag=apply_rag, 
        apply_reflection=apply_reflection, 
        reflect_iter=reflect_iter
    )

    batip.prepare_requests(
        template=template,
    )

    batip.init_pipe(
        diffusion_model=diffusion_model,
        hparam_name=hparam_name,
    )

    batip.generate_imgs(
        imgs_per_prompt=imgs_per_prompt,
        after_edit=False,
    )

    batip.eval_ratio(after_edit=False)

    batip.edit_concept(
        edit_iter=edit_iter,
        mom2_weight=mom2_weight,
        edit_weight=edit_weight,
        return_orig_text_model=return_orig_text_model,
        recompute_factors=recompute_factors,
        verbose=verbose,
    )
    
    batip.generate_imgs(
        imgs_per_prompt=imgs_per_prompt,
        after_edit=True,
    )

    batip.eval_ratio(after_edit=True)
