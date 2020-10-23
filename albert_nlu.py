import os
import torch
from transformers import AlbertTokenizer
from ts.torch_handler.base_handler import BaseHandler


class AlbertHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """
        self.tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
        properties = context.system_properties
        self.map_location = 'cpu'
        self.device = torch.device("cpu")
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # model def file
        model_file = self.manifest['model'].get('modelFile', '')

        if model_file:
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
        else:
            self.model = self._load_torchscript_model(model_pt_path)

        self.model.to(self.device)
        self.model.eval()
        # Load class mapping for classifiers
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentence = text
        print("Received text: '%s'", sentence)
        encoded = self.tokenizer.encode(sentence, add_special_tokens=True)
        print(encoded)
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

    def postprocess(self, data):
        entity_labels = " ".join([str(v) for v in data[1].squeeze(0).argmax(-1).tolist()])
        return [f"Intent: {str(data[0].argmax(-1).item())} Slots: {entity_labels}"]


