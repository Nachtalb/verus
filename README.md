# Verus

## Overview

Verus is a Python-based application that leverages deep learning models to
predict tags for images. It provides a daemon to run predictions as a service
and a client to interact with the daemon for tag predictions. The application
supports filtering and organising images based on their predicted tags. The name
"Verus" comes from the Latin word for "true" or "real", reflecting the tool's
purpose of uncovering the true characteristics of images.

## Installation

Ensure you have Python 3.11.9 installed. Clone the repository and install
dependencies using Poetry:

```sh
git clone <repository_url>
cd verus
poetry install
```

## Usage

### Running the Daemon

The daemon runs a prediction service that listens for incoming connections and
processes image predictions.

```sh
verusd [--host <host>] [--port <port>]
```

### Client Usage

The client can be used to send images to the daemon for prediction.

```python
from pathlib import Path
import numpy as np
from verus.client import PredictClient

client = PredictClient(host='localhost', port=65432)

# Predict from a file
image_path = Path('path/to/image.jpg')
prediction = client.predict(image=image_path, score_threshold=0.4)

# Predict from an ndarray
image_array = np.array([...])  # Your image data
prediction = client.predict(image=image_array, score_threshold=0.4)
```

### Command-Line Interface

The main entry point provides commands for identifying images and moving them
based on their tags.

```sh
verus identify --image <image_path> [--threshold <score_threshold>] [--save-json]
verus move <input_directory> <output_directory> [--threshold <score_threshold>] [--save-json] [--dry-run] [--unknown-dir <dir>] [--no-move-unknown]
```

### Configuration

- **tags.json**: Define tags and their filters.

## Example Tags Configuration (tags.json)

```json
{
  "vtuber": {
    "type": "or",
    "value": ["hololive", "nijisanji"]
  },
  "protagonist": "main_character",
  "chibi": {
    "type": "and",
    "value": [
      {
        "type": "regex",
        "value": "chibi"
      },
      "cute"
    ]
  },
  "sci-fi": {
    "type": "or",
    "value": [
      "robot",
      {
        "type": "and",
        "value": [
          {
            "type": "regex",
            "value": "cyber.*"
          },
          "future"
        ]
      }
    ]
  }
}
```

## License

This project is licensed under the LGPL 3.0 License. See the LICENSE file for
details.
