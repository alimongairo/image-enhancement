## Run instructions

### How to run pipeline
1. Clone repository
2. Download pretrained models from https://drive.google.com/drive/folders/1NFyBelhhlCcs35coB0Vjupwow56eYgwo?usp=sharing
3. Place NAFNet pretrained model in ./NAFNet/experiments/pretrained_models/
4. Place ShadowNet pretrained model in ./ShadowNet/model/
5. Run ``` python pipeline.py <input-image-path> ```
6. Result can be found in ./output/output.jpg


### How to run web-application
1. Clone repository
2. Install streamlit (instructions could be found here https://docs.streamlit.io/library/get-started/installation)
3. Open Anaconda terminal
4. Run ``` streamlit app.py ```
