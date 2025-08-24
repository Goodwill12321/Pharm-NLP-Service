This is a HTTP service with POST request that use model from Hugging Face to extract parts from non formalized pharm names

Install:
install python version less than 3.13 (sentencepiece library requeirments)

Run pip
pip install -r requirements.txt
or
"python.exe -m pip install -r requirements.txt" (for embedded version without docker)

or 
"docker build -t pharm-nlp ."

for running in docker 

Usage:
py .\pharm_nlp_service.py 

or (for docker)

docker run -p 8000:8000 pharm-nlp

Test from client:

curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d "@request.json"
(There is example of request.json  in repository)


