FROM public.ecr.aws/lambda/python:3.11

# Install build tools
RUN yum install -y gcc gcc-c++ make cmake

# Copy requirements and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Copy your function code
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]