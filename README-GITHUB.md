# a Simple Gemini Clone

## Introduction

This simple Gemini clone is a React-based chatbot inspired by Google Gemini. Initially, it was conceived as a straightforward clone, but this project evolved into an excercise in creative problem-solving and working with AWS in the face of the current Google Gemini API access restrictions in Canada.

## Features

- **Google Gemini-like Interface**: Recreates the intuitive and user-friendly interface of Google Gemini, offering users a seamless experience.
- **Cross-Region API Access**: Implements a custom AWS Lambda server in Node.js to bypass regional restrictions, enabling unrestricted API access.
- **Dynamic Data Retrieval**: Efficient communication with the AWS Lambda server to fetch and display data, ensuring real-time accuracy and responsiveness.

## Technologies

- **Frontend**: Developed using React with Vite for an optimized development and build process, leveraging its component-based architecture for a scalable and maintainable codebase.
- **Backend**: Utilizes Node.js to set up an AWS Lambda function that acts as a proxy, facilitating API requests that would otherwise be restricted in Canada.
- **API Integration**: Demonstrates adept handling of third-party API integration, ensuring seamless data flow despite geographical constraints.

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

- Install [Node.js](https://nodejs.org/en/) (which comes with [npm](http://npmjs.com)) on your computer.
  
### Installation

1. Clone the repository:
git clone (https://github.com/olyaiy/gemini-clone.git)

2. Navigate to the project directory:
cd gemini-clone

3. Install project dependencies using npm:
```npm install``` Or using yarn: ```yarn```

### Running the Application

1. To start the Vite development server:
```npm run dev``` Or if you are using yarn: ```yarn dev```

2. Open [http://localhost:3000](http://localhost:3000) to view it in the browser.
