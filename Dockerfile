# Stage 1: Build the React frontend
FROM node:18-alpine AS build

WORKDIR /app

# Copy package.json and package-lock.json
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend source code
COPY src ./src
COPY public ./public
COPY vite.config.js ./
COPY index.html ./

# Build the frontend
RUN npm run build

# Stage 2: Create the final image with the Python backend
FROM python:3.10-slim

WORKDIR /app

# Install backend dependencies
COPY server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY server/ ./server

# Copy the built frontend from the build stage
COPY --from=build /app/dist ./dist

# Expose the port the backend will run on
EXPOSE 7860

# Command to run the backend server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
