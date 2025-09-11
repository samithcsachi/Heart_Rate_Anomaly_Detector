# Use Python 3.13 base image 
FROM python:3.13-slim
 
#working directory 
WORKDIR /app




#copy
COPY requirements.txt .

#run
RUN pip install --no-cache-dir -r requirements.txt


#Copy rest of application code 
COPY . .



#Expose the application port 
EXPOSE 8000


# Command to start FastAPI application 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


