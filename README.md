# Dataset handling
How to load the dataset and convert the text file as a dataset to JSON
To access the dataset we can get our data from https://www.aminer.org/billboard/aminernetwork by downloading all three parts of the data now it's time to convert the dataset text file into JSON. To learn more about the conversion method please checkout text_to_json.py

# Hybrid Model
To run the Hybrid Modelv (Hybrid_model.py) after initializing the MongoDB and setting up the local host you need to change parameters in the main function to connect to the right local host and make sure the database names and collection name match the one in MongoDB

# Baseline 1 
To run baseline 1 which includes the content base and collaborative filtering baseline you need to run recommender.py.after initializing the MongoDB and setting up the local host you need to change parameters in the main function to connect to the right local host and make sure the database names and collection names match the ones in MongoDB
# Baseline 2
To run Baseline 2 which includes the graph citation network baseline you need to run modified_citation_network.py.after initializing the MongoDB and setting up the local host you need to change parameters in the main function to connect to the right local host and make sure the database names and collection names match the ones in MongoDB
