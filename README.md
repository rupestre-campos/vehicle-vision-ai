# VEHICLE VISION AI

This repo is a proof of concept to solve challenge to classify vehicle color, type and % occlusion
using high resolution satellite imagery using Artificail Inteligence models.


## Instalation

Install ollama
```
curl -fsSL https://ollama.com/install.sh | sh
```

Install and serve model gemma3:4b model, that is a vision and prompt model freely available

```
ollama run gemma3:4b &
```

Python libraries

```
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Run code to classify input
Create folder named 'data' and put inside the input folder containing .png and .txt input files,
or instead point the main.py script to desired folder.
Run with
```
python main.py
```

### To view full image with annotated boundinboxes
As part of the data exploratory analysis we may run script draw_anotated_image.py
in order to create a new image containing all detected rotated bounding boxes to check inputs.
