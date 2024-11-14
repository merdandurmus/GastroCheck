
# Gastro-Check

Gastro-Check is a Python-based application designed for tracking and evaluating Gastro-Intestinal (GI) tract imagery. This project integrates machine learning models to analyze GI images, facilitating real-time digit recognition and tracking of relevant data for medical analysis.

## Project Structure

- **GI-Tract-Images**: Contains image data of the GI tract used for training, testing, or evaluation purposes.
- **Gastro-Check-App**: The main application directory for Gastro-Check, containing all core scripts, including tracking and AI evaluation functionalities.
- **README.md**: Documentation and overview of the project.
- **requirements.txt**: List of dependencies required to run the project.

## Features

- **Real-Time Digit Recognition**: Utilizes a Convolutional Neural Network (CNN) to perform real-time recognition of digits related to medical imagery.
- **GI Tract Tracking**: Provides live tracking of motion metrics for GI tract analysis.
- **Image-Based Analysis**: Uses machine learning to analyze colored patterns and features within GI images.
- **Custom Data Visualization**: Displays tracked data metrics like time, path length, angular length, response orientation, and more.
- **User-Friendly Interface**: Intuitive GUI for medical professionals to use the app without technical barriers.

## Getting Started

### Prerequisites
To run the project, ensure you have Python installed along with the necessary dependencies listed in `requirements.txt`.

### Installation
Clone the repository:
```bash
git clone https://github.com/your-username/Gastro-Check.git
cd Gastro-Check
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the application
Navigate to the main application folder:
```bash
cd Gastro-Check-App
```

Then execute the main script:
```bash
python main.py
```

## Usage

1. **Load Images**: Place images for evaluation in the `GI-Tract-Images` folder.
2. **Start Tracking**: Use the application's GUI to begin tracking and analysis.
3. **Evaluate AI Model**: The app includes a script to evaluate the AI model's performance on the GI images. Use `evaluateAI.py` for this purpose.

## Development

### Training the Model
To train or retrain the AI model used for analysis:

1. Prepare your dataset and place it in the `GI-Tract-Images` directory.
2. Modify the training parameters if necessary.
3. Run the training script provided in the `Gastro-Check-App` folder.

### Testing
For testing the trained model's performance, run the following command:
```bash
python evaluateAI.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`feature-xyz`).
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgements

- Thank you to all contributors who have supported the development of this application.
- Special mention to any open-source libraries used in this project.

## Contact

For any questions or feedback, please contact Merdan at [your-email@example.com].
