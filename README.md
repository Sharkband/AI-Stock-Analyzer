# AI-Stock-Analyzer
This app creates AI Stock models so you can predict stock prices in the next 30 days.

demo:

https://youtu.be/jYVqolnyd4M


to run using docker:

 # build image
docker build -t ai-stock-analyzer .

# Run the container
docker run -p 8000:8000 ai-stock-analyzer

then go to http://localhost:8000/


without docker run API:

pip install -r requirements.txt

cd backend

python api/api.py

without docker run React frontend:

in a new terminal

cd frontend/src/react

npm install

npm run dev





