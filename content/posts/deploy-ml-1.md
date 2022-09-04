---
title: "🚀 Deploy ML model: Part 1 - Running triton server"
date: 2022-08-24T23:46:30+04:00
draft: true
---

In this series, I will tell you how to deploy your ML model to the real production. 

* 📖 All code used in this article is available on [GitHub](https://github.com/alxmamaev/deploy_ml_course).
* ⭐️ If you like this course, please star it on [GitHub](https://github.com/alxmamaev/deploy_ml_course). 
* ❓ If you have any problems, suggestions or questions, you may create [issue](https://github.com/alxmamaev/deploy_ml_course/issues).

# Why this article exist?

If you found this article, you probably saw a lot of tutorials _"How to deploy your ml model in 5 minutes"_, mostly this tutorial uses Flask as API backend which just call PyTorch model, so actually that is not a best way, it may bee good anought for quick-and-dirty developement, for example hackathons or MVP, but for a real production **it is really bad**, because this solution does not support requests batching, multiple-instance inference, scaling and so on. 

In this series i will tell you a **better way to deploy models**, which i got from my expirience in machine learning. 



# Syllabus
* **Run Triton server** (This article)
* ~~Data preprocessing pipeline~~ (Work in progress..) 
* ~~Request batching~~ (Work in progress..)
* ~~Testing~~ (Work in progress..)
* ~~Performance testing~~ (Work in progress..)
* ~~Docker Swarm~~ (Work in progress..)
* ~~Running model on multiple nodes~~ (Work in progress..)
* ~~C++ preprocessing module in Triton~~ (Work in progress..)
* ~~TensorRT~~ (Work in progress..)
* ~~Model quantization~~ (Work in progress..)

# Introduction
In this series of articles we will deploy **PyTorch** model into the production. Our final server will able to process at least **200 users** at same time. 

For the demo-model Ix choused a [Twitter roBERTa sentiment classifier](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) from HuggingFace Hub. I choused NLP model, because this model is really big (400MB weights), NLP is quite popular area, but there are not much tutorials about it compare to Computer Vision. 

So if you have computer vision task, such image/video classification, or vidio generation, it's still helpfull to you read this series, because basics things are the same with model-specific changes, like model conversion or preprocessing. The same for other frameworks like TensorFlow or non-neural models such XGBoost. 

Maybe in future i will add some examples for other frameworks and tasks, so ⭐️ GitHub repo to make it real.


# TorchScript

On HuggingfaceHub you can find example how to use model, it is quite simple.

*Hint:* Install `transformers` library before run this code.

```python
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

labels = ["negative", "neutral", "positive"]

model_name = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()

text = "Good night 😊"
encoded_input = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    output = model(**encoded_input)

label_id = torch.argmax(output[0])
print(labels[label_id])
```

This code will print you `positive` that is classification result for the text `Good night 😊`.
There are we have a `model` which is `torch.nn.Module` and tokenizer,  which preprocess text into squence of tokens and attention mask.


Using `torch.nn.Module` is good for model training, validation or small tests, but for production Torch model is overkill, because it's contains a dynamic computation graph with lot of funtionality for passong gradients. 

In production we does not need model training actually, and our model does not need in-flight changes, so we may use a static-compiled computation graph which can make only forward pass. So here the **TorchScript** takes the stage.

### Docker
Before we are starting a model conversion i want to say few words about docker.
I highly recommended you to isolate all your pipeline stages into **docker containers**.

**Why I should use docker?**
* It's separate your developement enviroment, so if you have some requiremtns conflict in different stages, it's will not break your pipeline
* You may work on different machine with the same enviroment. For example you may run the same docker container on your laptop or server and you may expect the same behaviour.
* It's easy to reproduce. Docker container freeze your enviroment include a binaries which are you using, operation system and so on, this is not possible using only **virtualenv** or **conda**.

**I will not exaplain how to work with docker in this article, you may read more about it on the Interenet.**

So i created directory in my project named `docker`, where i can store docker files for stages of my pipeline such is: *model conversion, inference server, server client, server tests*


Let's create dockerfile for the first stage of pipeline - model torchscript conversion.


*docker/torchscript_conversion.Dockerfile* :
```dockerfile
FROM nvcr.io/nvidia/pytorch:22.07-py3

RUN pip3 install transformers
```

Now i can build and run it:
```bash
docker build . \
    -f docker/torchscript_conversion.Dockerfile \
    -t torchscript_conversion:latest

docker run --rm -it -v $(pwd):/deploy_ml torchscript_conversion:latest bash
```

If you are already fammiliar with a Docker, i highly recommend use [VSCode with remote developement extension](https://code.visualstudio.com/docs/remote/containers), to work write and run code inside docker directly.


### What is TorchScript

**TorchScript** -- is optimised graph format for the model inference introduced by PyTorch in 2019. This format allows to convert out model into single file which contains a model weights, computation graph, so we does not need a model code for inference. This model can be loaded outside of python, for example in C++ or Java.

For the model conversion I will use **torch.jit.trace** function which convert model into the static graph. In fact, converter just make a forward pass for the model with some *dummy inputs* and remember sequence of operations which be used to transform this input into final model prediction. 

```python
traced_model = torch.jit.trace(model, (input_tensor_1, input_tensor_2))
```
The **trace** function expect that model will have one or multiple **torch.Tensor** arguments at input and one or mode **torch.Tensor** on output. So it's means that **key-arguments** or arguments which is not a **torch.Tensor** is not allowed to the input, and output may be only Tensor or tuple of Tensors.


### Conversion adapter
In our case `model` expects two key-argument inputs: `input_ids` (which contains encoded sequence of tokens) and `attention_mask`, output provides a dict-like structure witch logits and some other spesific data. This model cannot be converted into TorchScript. So we need to make some *adapter*, which make conversion possible. 

Thi is code of my simple conversion adapter:
```python
import torch

class ModelTorchTraceAdapter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(intpu_ids=input_ids, 
                          attention_mask=attention_mask).logits
```
This code just get classig arguments and pass it to model as key-args, and also take a logits key from returned output. So now, this model can be eseally converted.

> I put this sctipt into `utils/adapter_torchscript.py`

### Converting model
The conversion code is quite simple too:
```python
def main(args):
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    vocab_size = tokenizer.vocab_size

    dummy_input_ids = torch.randint(0, vocab_size, (1, 100))
    dummy_mask = torch.ones_like(dummy_input_ids)

    adapted_model = ModelTorchTraceAdapter(model)
    traced_model = torch.jit.trace(adapted_model, (dummy_input_ids, dummy_mask))

    torch.jit.save(traced_model, args.output)
```
This script creates a dummy input for conversion, which contain fake data and then call **trace** function. It will return me traced **TorchScript** model, which I can save.

So I little bit improved this script: add logging and testing of the converted model.

```python
def main(args):
    logging.info("Loading model and tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    vocab_size = tokenizer.vocab_size
    logging.info(f"Model vocab size is: {vocab_size}")

    dummy_input_ids = torch.randint(0, vocab_size, (1, 100))
    dummy_mask = torch.ones_like(dummy_input_ids)

    logging.info(f"Tracing model")
    adapted_model = ModelTorchTraceAdapter(model)
    traced_model = torch.jit.trace(adapted_model, (dummy_input_ids, dummy_mask))

    test_inputs = [
        "I love you so much!",
        "I hate this movie",
        "Ok, let's see tomorrow."
    ]

    logging.info(f"Checking model outputs")
    check_fn = partial(compare_models, model, traced_model, tokenizer)

    for test_id, text in enumerate(test_inputs):
        assert check_fn(text), f"Error on text \"{text}\""
        logging.info(f"Test:{test_id} OK")

    logging.info(f"Conversion is done")
    logging.info(f"Model saved to {args.output}")
    torch.jit.save(traced_model, args.output)
```

Let's run the `torchscript_conversion` docker container and then run the conversion torch script. 

```bash
cd /deploy_ml
python3 convert_torchscript.py \
--model_name cardiffnlp/twitter-roberta-base-sentiment \
--output traced_sentiment_classifier.pt
```

*Script output:*
```bash
INFO:root:Loading model and tokenizer
Downloading config.json: 100%|█████████| 747/747 [00:00<00:00, 203kB/s]
Downloading pytorch_model.bin: 100%|█████████| 476M/476M [01:50<00:00, 4.50MB/s]
Downloading vocab.json: 100%|█████████| 878k/878k [00:01<00:00, 735kB/s]
Downloading merges.txt: 100%|█████████| 446k/446k [00:00<00:00, 761kB/s]
Downloading special_tokens_map.json: 100%|█████████| 150/150 [00:00<00:00, 48.8kB/s]
INFO:root:Model vocab size is: 50265
INFO:root:Tracing model
INFO:root:Checking model outputs
INFO:root:Test:0 OK
INFO:root:Test:1 OK
INFO:root:Test:2 OK
INFO:root:Conversion is done
INFO:root:Model saved to traced_sentiment_classifier.pt
```

So, now we have a converted TorchScript model which we can run on some server. We will not write our-own non-efficient server using Flask or FastAPI, instead of it we will run a fast and efficient **Triton Server** which is maintained by Nvidia.  

# Triton server

```bash
mkdir -p model_repository/sentiment_classifier
touch model_repository/sentiment_classifier/config.pbtxt

mkdir -p model_repository/sentiment_classifier/1 
cp traced_sentiment_classifier.pt model_repository/sentiment_classifier/1/model.pt
```

```
model_repository
└── sentiment_classifier
    ├── 1
    │   └── model.pt
    └── config.pbtxt
```

```protobuf
name: "sentiment_classifier"
backend: "pytorch"
max_batch_size: 1

input [
    {
        name: "INPUT__0"
        data_type: TYPE_INT64
        dims: [ -1 ]
    },
    {
        name: "INPUT__1"
        data_type: TYPE_INT64
        dims: [ -1 ]
    }
]
output [
    {
        name: "OUTPUT__0"
        data_type: TYPE_FP32
        dims: [ 3 ]
    }
]
```

```dockerfile
FROM nvcr.io/nvidia/tritonserver:22.07-pyt-python-py3 

# We will add other dependencies here
```

```bash
docker build . -f docker/triton.Dockerfile -t tritonserver:latest
```
```bash
docker run \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/model_repository:/model_repository \
    tritonserver:latest \
    tritonserver --model-repository /model_repository
```

```
I0825 20:47:37.672667 1 server.cc:586]
+---------+---------------------------------+---------------------------------+
| Backend | Path                            | Config                          |
+---------+---------------------------------+---------------------------------+
| pytorch | /opt/tritonserver/backends/pyto | {"cmdline":{"auto-complete-conf |
|         | rch/libtriton_pytorch.so        | ig":"true","min-compute-capabil |
|         |                                 | ity":"6.000000","backend-direct |
|         |                                 | ory":"/opt/tritonserver/backend |
|         |                                 | s","default-max-batch-size":"4" |
|         |                                 | }}                              |
|         |                                 |                                 |
+---------+---------------------------------+---------------------------------+

I0825 20:47:37.672927 1 server.cc:629]
+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| sentiment_classifier | 1       | READY  |
+----------------------+---------+--------+

I0825 20:47:37.673253 1 tritonserver.cc:2176]
+----------------------------------+------------------------------------------+
| Option                           | Value                                    |
+----------------------------------+------------------------------------------+
| server_id                        | triton                                   |
| server_version                   | 2.24.0                                   |
| server_extensions                | classification sequence model_repository |
|                                  |  model_repository(unload_dependents) sch |
|                                  | edule_policy model_configuration system_ |
|                                  | shared_memory cuda_shared_memory binary_ |
|                                  | tensor_data statistics trace             |
| model_repository_path[0]         | /model_repository                        |
| model_control_mode               | MODE_NONE                                |
| strict_model_config              | 0                                        |
| rate_limit                       | OFF                                      |
| pinned_memory_pool_byte_size     | 268435456                                |
| response_cache_byte_size         | 0                                        |
| min_supported_compute_capability | 6.0                                      |
| strict_readiness                 | 1                                        |
| exit_timeout                     | 30                                       |
+----------------------------------+------------------------------------------+

I0825 20:47:37.679998 1 grpc_server.cc:4608] Started GRPCInferenceService at 0.0.0.0:8001
I0825 20:47:37.680359 1 http_server.cc:3312] Started HTTPService at 0.0.0.0:8000
I0825 20:47:37.729232 1 http_server.cc:178] Started Metrics Service at 0.0.0.0:8002
```


```bash
curl -X POST localhost:8000/v2/repository/index
```

```json
[{"name":"sentiment_classifier","version":"1","state":"READY"}]
```

```bash
docker run \
    -d \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/model_repository:/model_repository \
    tritonserver:latest \
    tritonserver --model-repository /model_repository
```

```bash
docker ps
```
```
CONTAINER ID   IMAGE                 COMMAND                  CREATED         STATUS         PORTS                                                           NAMES
54c81a1ff1e2   tritonserver:latest   "/opt/nvidia/nvidia_…"   3 minutes ago   Up 3 minutes   0.0.0.0:8000-8002->8000-8002/tcp, :::8000-8002->8000-8002/tcp   suspicious_bor
```

```bash
docker kill 54c81a1ff1e2
```

# Triton client

```
mkdir client
touch client/client.py
touch docker/client.Dockerfile
```

```
from argparse import ArgumentParser
import numpy as np
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer


def parse():
    parser = ArgumentParser()
    parser.add_argument("--triton_adress")
    parser.add_argument("--model_name")
    parser.add_argument("--input_text")

    return parser.parse_args()


def main(args):
    inputs = []
    outputs = []

    triton_client = grpcclient.InferenceServerClient(args.triton_adress, verbose=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    encoded_input = tokenizer(args.input_text, return_tensors="np")
    input_lenghts = encoded_input["input_ids"].shape[1]

    inputs.append(grpcclient.InferInput("INPUT__0", [1, input_lenghts], "INT64"))
    inputs.append(grpcclient.InferInput("INPUT__1", [1, input_lenghts], "INT64"))
    outputs.append(grpcclient.InferRequestedOutput("OUTPUT__0"))

    inputs[0].set_data_from_numpy(encoded_input["input_ids"].astype(np.int64))
    inputs[1].set_data_from_numpy(encoded_input["attention_mask"].astype(np.int64))

    results = triton_client.infer(model_name="sentiment_classifier", inputs=inputs, outputs=outputs)
    logits = results.as_numpy("OUTPUT__0")

    class_index = np.argmax(logits[0])
    labels = ["negative", "neutral", "positive"]

    print("Predict:", labels[class_index])


if __name__ == "__main__":
    args = parse()
    main(args)
```

```
docker build . -f docker/client.Dockerfile -t tritonclient:latest
```

```
docker run --net=host --rm tritonclient:latest --triton_adress=127.0.0.1:8001 --input_text="I love you" --model_name=cardiffnlp/twitter-roberta-base-sentiment
```

```
Downloading config.json: 100%|██████████| 747/747 [00:00<00:00, 212kB/s]
Downloading vocab.json: 100%|██████████| 878k/878k [00:00<00:00, 955kB/s]
Downloading merges.txt: 100%|██████████| 446k/446k [00:00<00:00, 595kB/s]
Downloading special_tokens_map.json: 100%|██████████| 150/150 [00:00<00:00, 39.6kB/s]
Predict: positive
```

Simple improovment
```dockerfile
RUN python3 -c 'from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")'
```

```bash
docker run --net=host --rm tritonclient:latest --triton_adress=127.0.0.1:8001 --input_text="I love you"
```

Output:
```
Predict: positive
```