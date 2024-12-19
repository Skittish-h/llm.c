import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(prompt, max_length=64):
    tokenizer = AutoTokenizer.from_pretrained("converted_model")
    model = AutoModelForCausalLM.from_pretrained("converted_model", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
    attention_mask = inputs["attention_mask"]
    print(inputs["input_ids"])
    outputs = model.generate(
        inputs["input_ids"], 
        # attention_mask=attention_mask, 
        max_length=max_length, 
        do_sample=False, 
        #top_p=0.95, 
        #top_k=50, 
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = 'Hello, I am Elon Musk and I like'
    response = generate_text(prompt)
    print(response)

    # print("Chatbot: Hi! How can I help you today?")
    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() in ["exit", "quit", "stop"]:
    #         print("Chatbot: Goodbye!")
    #         break
    #     response = generate_text(user_input)
    #     print(response)