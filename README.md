# Deploying BERT NER Models on IBM Watson Machine Learning with Hugging Face Transformers


To deploy NER (Named Entity Recognition) models using BERT transformers from Hugging Face on IBM Watson Machine Learning (WML) with `ibm-watsonx-ai`, you can follow the steps below. Here, I provide a full Python code to perform this activity using a standard NER BERT model from Hugging Face.

### Step-by-Step Guide and Full Python Code

1. **Set up the environment**
2. **Create model definition**
3. **Train the model**
4. **Persist the trained model**
5. **Deploy and score**
6. **Clean up**
7. **Summary and next steps**

### 1. Set Up the Environment

#### Install and Import the `ibm-watsonx-ai` and Dependencies

```python
!pip install wget | tail -n 1
!pip install -U ibm-watsonx-ai | tail -n 1
!pip install transformers | tail -n 1
!pip install torch | tail -n 1
```

#### Authenticate the Watson Machine Learning Service

```python
from ibm_watsonx_ai import Credentials, APIClient

username = 'PASTE YOUR USERNAME HERE'
api_key = 'PASTE YOUR API_KEY HERE'
url = 'PASTE THE PLATFORM URL HERE'

credentials = Credentials(
    username=username,
    api_key=api_key,
    url=url,
    instance_id="openshift",
    version="5.0"
)

client = APIClient(credentials)
```

#### Set Up the Space

```python
space_id = 'PASTE YOUR SPACE ID HERE'
client.set.default_space(space_id)
```

### 2. Create Model Definition

#### Prepare Model Definition Metadata

```python
model_definition_metadata = {
    client.model_definitions.ConfigurationMetaNames.NAME: "BERT NER Model",
    client.model_definitions.ConfigurationMetaNames.DESCRIPTION: "BERT model for Named Entity Recognition",
    client.model_definitions.ConfigurationMetaNames.COMMAND: "ner_train.py",
    client.model_definitions.ConfigurationMetaNames.PLATFORM: {"name": "python", "versions": ["3.11"]},
    client.model_definitions.ConfigurationMetaNames.VERSION: "1.0",
    client.model_definitions.ConfigurationMetaNames.SPACE_UID: space_id
}
```

#### Get Sample Model Definition Content File

```python
import wget, os

filename = 'bert-ner-model.zip'

if not os.path.isfile(filename):
    filename = wget.download('URL_TO_YOUR_ZIP_FILE_CONTAINING_MODEL_DEFINITION')

!unzip -oqd . bert-ner-model.zip
```

#### Publish Model Definition

```python
definition_details = client.model_definitions.store(filename, model_definition_metadata)
model_definition_id = client.model_definitions.get_id(definition_details)
print(model_definition_id)
```

### 3. Train the Model

#### Prepare Training Metadata

```python
training_metadata = {
    client.training.ConfigurationMetaNames.NAME: "BERT NER Training",
    client.training.ConfigurationMetaNames.DESCRIPTION: "Training BERT model for Named Entity Recognition",
    client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
        "name": "NER results",
        "connection": {},
        "location": {"path": f"spaces/{space_id}/assets/experiment"},
        "type": "fs"
    },
    client.training.ConfigurationMetaNames.MODEL_DEFINITION: {
        "id": model_definition_id,
        "hardware_spec": {"name": "K80", "nodes": 1},
        "software_spec": {"name": "pytorch-onnx_rt24.1-py3.11"}
    },
    client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [
        {
            "name": "training_input_data",
            "type": "fs",
            "connection": {},
            "location": {"path": "bert-ner-dataset"},
            "schema": {"id": "idmlp_schema", "fields": [{"name": "text", "type": "string"}]}
        }
    ]
}
```

#### Train the Model

```python
training = client.training.run(training_metadata)
```

#### Get Training ID and Status

```python
training_id = client.training.get_id(training)
print(training_id)
client.training.get_status(training_id)['state']
```

#### Get Training Details

```python
import json

training_details = client.training.get_details(training_id)
print(json.dumps(training_details, indent=2))
```

### 4. Persist the Trained Model

#### Publish the Model

```python
software_spec_id = client.software_specifications.get_id_by_name('pytorch-onnx_rt24.1-py3.11')

model_meta_props = {
    client.repository.ModelMetaNames.NAME: "BERT NER Model",
    client.repository.ModelMetaNames.TYPE: "pytorch-onnx_2.1",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: software_spec_id
}

published_model_details = client.repository.store_model(training_id, meta_props=model_meta_props)
model_id = client.repository.get_model_id(published_model_details)
```

### 5. Deploy and Score

#### Create Online Deployment

```python
deployment = client.deployments.create(
    model_id, meta_props={
        client.deployments.ConfigurationMetaNames.NAME: "BERT NER Deployment",
        client.deployments.ConfigurationMetaNames.ONLINE: {}
    }
)

scoring_url = client.deployments.get_scoring_href(deployment)
deployment_id = client.deployments.get_id(deployment)
```

#### Get Deployment Details

```python
deployments_details = client.deployments.get_details(deployment_id)
print(json.dumps(deployments_details, indent=2))
```

#### Score Deployed Model

Prepare sample scoring data:

```python
from transformers import pipeline

# Load pre-trained model and tokenizer
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Sample text for testing
test_text = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge."

# Get NER predictions
ner_results = ner_pipeline(test_text)
print(ner_results)
```

### 6. Clean Up

Refer to the sample notebook for detailed clean-up steps.

### 7. Summary and Next Steps

You have successfully deployed a BERT NER model using Hugging Face transformers on IBM Watson Machine Learning. You learned how to set up the environment, create model definitions, train the model, persist the trained model, and deploy and score the model. Check out the [Online Documentation](https://ibm.github.io/watsonx-ai-python-sdk/samples.html) for more samples, tutorials, and documentation.

### Author

**Ruslan Magana Vsevolodovna**, IBM CIC Italy

### License

This notebook and its source code are released under the terms of the MIT License.