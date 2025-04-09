import logging, os, time
from openai import OpenAI

from LAMB.lamb.utils import file_to_string
# from utils import file_to_string

class LAMB:
    def __init__(self, openai_client, prompt_dir, log_dir, mem_dir):
        self.openai_client = openai_client

        self.prompt_dir = prompt_dir

        self.log_dir = log_dir
        self.mem_dir = mem_dir

    def get_user_input(self):
        user_input = input("Enter your textual prompt for image generation: ")
        logging.info(f"User input: {user_input}")
        with open (f"{self.mem_dir}/chat_history.txt", "a") as f:
            f.write(f"User input: {user_input}\n")
        return user_input

    def integrate_prompts(self, user_input, prev_response='', apply_rag=False, apply_reflection=False):
        initial_system = file_to_string(f'{self.prompt_dir}/system/initial_system.txt')
        background_knowledge = file_to_string(f'{self.prompt_dir}/docs/background_knowledge.txt')
        response_signature = file_to_string(f'{self.prompt_dir}/system/response_signiture.txt')

        system_content = initial_system.format(background_knowledge=background_knowledge,
                                               response_signature=response_signature)

        if apply_reflection:
            initial_user = file_to_string(f'{self.prompt_dir}/user/initial_user_reflect.txt')
            cot_feedback = file_to_string(f'{self.prompt_dir}/reflection/cot_feedback.txt')
            output_feedback = file_to_string(f'{self.prompt_dir}/reflection/bias_feedback.txt')

            user_content = initial_user.format(user_input=user_input,
                                               prev_response=prev_response,
                                               cot_feedback=cot_feedback,
                                               output_feedback=output_feedback)
        
        else:
            initial_user = file_to_string(f'{self.prompt_dir}/user/initial_user_vani.txt')
            few_shots = file_to_string(f'{self.prompt_dir}/few_shots/examples.txt')
            
            user_content = initial_user.format(user_input=user_input,
                                               few_shots=few_shots)

        # chat_history = file_to_string(f'{self.mem_dir}/chat_history.txt')
        # log_history = file_to_string(f'{self.log_dir}/lamb.log')
        
        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}, {"role": "assistant", "content": prev_response}]
        
        return messages
    
    def predict_bias(self, model, messages):
        response_cur = None
        for attempt in range(1000):
            try:
                response_cur = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                break
            except Exception as e:
                logging.warning(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
        if response_cur is None:
            logging.warning(f"Iteration {iter}: Code terminated due to too many failed attempts!")
            exit()
        
        response = response_cur.choices[0].message.content
        logging.debug(f"GPT Output:\n " + response + "\n")
        with open (f"{self.mem_dir}/chat_history.txt", "a") as f:
            f.write(f"========================== GPT Output ==========================\n")
            f.write(response + "\n")
        prompt_tokens = response_cur.usage.prompt_tokens
        completion_token = response_cur.usage.completion_tokens
        logging.info(f"Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_token}")

        return response, prompt_tokens, completion_token
    
    def extract_attributes(self, response):
        user_input = response.split("**User input**: ")[1].split("\n")[0]
        bias = response.split("**Potential bias in adjectives**: ")[1].split("\n")[0]
        source2dest_text = response.split("**Variance of entities with the combination of adjectives**: ")[1].split("\n")[0]
        var = {"user_input": "", "bias": {}, "source2dest": {}}
        exec(f"var['user_input'] = '{user_input}'")
        exec(f"var['bias'] = {bias}")
        exec(f"var['source2dest'] = {source2dest_text}")
        return var['user_input'], var['bias'], var['source2dest']
    
    def prepare_requests(self, source2dest, template, seed=2025):
        requests = []
        for entity in source2dest:
            request = {
                "prompts": template, 
                "seed": seed,
                "source": entity,
                "dests": source2dest[entity],
            }
            requests.append(request)

        return requests

        

if __name__ == '__main__':
    timestamp = time.strftime('%Y%m%d-%H%M%S')

    PROMPT_DIR = 'LAMB/prompts'
    LOG_DIR = f'LAMB/results/{timestamp}/logs'
    MEM_DIR = f'LAMB/results/{timestamp}/memories'

    for dir in [LOG_DIR, MEM_DIR]:
        os.makedirs(dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        filename=f"{LOG_DIR}/lamb.log",
                        filemode='w')
    
    OPENAI_CLIENT = OpenAI(base_url = "",
                           api_key = "")
    
    model = "gpt-4o-2024-05-13"
    apply_rag = False
    apply_reflection = False
    reflect_iter = 1

    lamb = LAMB(OPENAI_CLIENT, PROMPT_DIR, LOG_DIR, MEM_DIR)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    # user_input = lamb.get_user_input()
    # user_input = "A painting of a farmer."
    # user_inputs = ["an attendant", "a cashier", "a nurse", "a driver"]
    user_inputs = ["a nurse"]
    template = ["a photo of {}",
                "an image of {}",
                "{}"]
    global_seed = 2025
    for user_input in user_inputs:
        logging.info(f"User input: \"{user_input}\"")
        messages = lamb.integrate_prompts(user_input, apply_rag=apply_rag, apply_reflection=False)
        response, prompt_tokens, completion_token = lamb.predict_bias(model, messages)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_token

        if apply_reflection:
            for iter in range(reflect_iter):
                messages = lamb.integrate_prompts(user_input, prev_response=response, apply_reflection=True)
                response = lamb.predict_bias(model, messages)
        
        try:
            user_input, bias, source2dest = lamb.extract_attributes(response)
            logging.info(f"User input: {user_input}")
            logging.info(f"Potential bias in adjectives: {bias}")
            logging.info(f"Variance of entities with the combination of adjectives: {source2dest}")
        except:
            logging.warning("Attrubutes extraction error!")
        
        requests = lamb.prepare_requests(source2dest, template, global_seed)
        logging.info(f"Requests: {requests}")
    
    logging.info(f"Total Prompt Tokens: {total_prompt_tokens}, Total Completion Tokens: {total_completion_tokens}")
