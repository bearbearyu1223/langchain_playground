## On-Device QnA with LangChain and Llama2
Third-party commercial large language model (LLM) providers, such as OpenAI's GPT-4 and Google Bard, have enabled widespread access to LLM capabilities through easy API integration. Nonetheless, certain situations driven by concerns related to data privacy and compliance, may necessitate the deployments of these models on-device.  

In this sample project, we will explore the process of running quantized editions of open-source Llama2 models on local CPUs for Retrieval Augmented Generation (RAG).
### Introduction 
#### Llama2
[Llama 2](https://ai.meta.com/resources/models-and-libraries/llama/), released by Meta, was pretrained on publicly available online data sources comprising 2 trillion tokens with a context length of 4096. The subsequent fine-tuned model, referred to as Llama-2-chat, underwent refinement through the incorporation of over 1 million human annotations, specifically tailored for chat use cases. Meta has extended accessibility to Llama 2 across a broad spectrum of users, encompassing individual developers, content creators, researchers, and businesses. This initiative aims to cultivate an environment conducive to responsible experimentation, innovation, and the scalable implementation of diverse ideas, thereby further democratizing Generative AI.

Llama 2 comes in a range of parameter sizes — 7B, 13B, and 70B — as well as pretrained and fine-tuned variations. 
#### LangChain 
#### Retrieval Augmented Generation
See the previous post at [Food QnA Chatbot : Help Answer Food Related Questions from Your Own Cookbook](https://bearbearyu1223.github.io/chatbot/2023/07/31/food-qna-on-server-llm.html) as a brief into to RAG. 
#### Running Llama2 on Local Machine 
Apple M1 Pro

### An Example Project 


### Instruction to run the example project 