import pathlib
import uuid
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

app = FastAPI()


@app.get('/')
def root():
    return {'message': 'hello, welcome to the main page of the api!', 'status': 400}


@app.get('/{file}')
def getData(file: str):
    # move one dir up
    one_dir_up = pathlib.Path(pathlib.Path.cwd()).parents[0]
    # get joined filepaths
    filepath = pathlib.Path.joinpath(one_dir_up, file)
    if pathlib.Path.exists(filepath):
        return FileResponse(path=filepath, media_type='application/octet-stream', filename=file)
    else:
        return {'status': 404, 'message': 'file does not exist'}


uvicorn.run('main:app', host='0.0.0.0', port=8080)
