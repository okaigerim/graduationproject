# Graduation Project

<p>In this repository, I have incorporated the practical component of my undergraduate thesis, which represents the culmination of my bachelor's degree. In accordance with the ruling of the Attestation Commission dated June 19, 2023, as recorded in protocol #202, I have been awarded an outstanding grade of 97 out of 100 (A+). </p>

**Files:** <br />
alakv-6.xlsx - Dataset <br />
alakvanalysis1.ipynb - Data Analysis Jupyter notebook <br />
app.py - Flask file for integration with ML model <br />
geocoder.py - geocoding coordinates to locations and vice versa <br />

<hr/>
<h1>Development of an information system for predicting real estate prices using machine learning and neural networks</h1>

### Tools for machine learning algorithms development:
<ul>
  <li>Web Scraper Google Chrome Extension for scraping data;</li>
  <li>Python programming language for data analysis and web development;</li>
  <li>Jupyter Notebook for machine learning model development;</li>
  <li>Numpy, Pandas, Scikit-learn, Keras, Pickle libraries of Python for data analysis and model training/testing.</li>
</ul>

### Tools for web application development:
<ul>
  <li>Pycharm IDE;</li>
  <li>HTML, CSS, JS, Bootstrap for front-end development;</li>
  <li>Flask framework for machine learning model deployment;</li>
  <li>The Google Sheets Geocode Awesome extension for geocoding latitude and longitude of real estate.</li>
</ul>
<h2>Data Preparation</h2>

<p>The first and most crucial stage of machine learning is data collection. Data collection is the process of collecting and measuring information on variables of interest in a defined systematic manner that allows answers to research questions, testing hypotheses, and evaluating results. The research data collection component is common to all fields of study, including physical and social sciences, humanities, business, and more. Regardless of the field of study or the preference for defining the data (quantitative, qualitative), accurate data collection is important to maintain the integrity of the study. The selection of appropriate data collection tools (existing, modified, or newly developed), as well as well-defined guidelines for their proper use, also reduces the likelihood of errors.</p>
<p>Consequences of incorrectly collected data include:</p>
<un>
<li>inability to answer research questions accurately;</li>
<li>inability to repeat and confirm research;</li>
<li>falsified results leading to loss of resources.</li>
</un><br/>
<p>In the process of data collection, it is very important to take into account possible errors and coding errors to be collected in the database in uniform units of measurement, and correct writing. The most important purpose of data collection is to ensure that rich information and reliable data are collected for analysis. The quality of the forecasting model is directly related to the quality of the data.</p>
<p>The website Krisha.kz is chosen as the main data provide for this project. Krisha.kz is a real estate website in Kazakhstan that provides information and services for those interested in buying, selling, or renting properties in the country. It offers a platform for real estate professionals and private individuals to list properties and reach potential buyers or renters. The website also provides a variety of tools and resources for real estate research and analysis, including property listings, property valuations, real estate news and market trends, and more. Krisha.kz aims to make the process of buying, selling, or renting real estate in Kazakhstan easier, more efficient, and more accessible for all users.</p>
<p>Extracting data from Krisha.kz is possible using a web-scraping tool.</p>
<p>First, the web scraper will be given one or more URLs to load before scraping. Then, the scraper loads the entire HTML code for the page in question. More advanced scrapers will render the whole website, including CSS and Javascript elements. Most web scrapers will output data to a CSV or Excel spreadsheet.</p>
<p>The web scraping tool is called WebScraper. Its form comes as a Chrome extension. Data preparation steps are described as follows: </p>
<ol>
<li>installation of WebScraper Chrome extension;</li>
<li>opening Krisha.kz and choosing apartments/buildings/lands/offices in the city of Almaty in the search bar;</li>
<li>opening WebScraper in the Developer Tools;</li>
<li>creating sitemap alakv; </li>
<li>as shown in the figure 1, pagination was chosen, which is each webpage the data will be extracted from; </li><br />
  
 ![image](https://github.com/okaigerim/graduationproject/assets/64651950/79713552-3e5c-4967-86b1-5f1076ae3b6f) <br />
 
<p><em>Figure 1. Choosing pagination</em></p>

<p><b>Pagination</b> is navigating through multiple pages to extract data.</p>
<li>creating element card with child elements such as name, price and address of each property, as shown in the figure 2; </li>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/8018c8ed-78fe-40ce-81b1-f803d42a0afc) <br />

<p><em>Figure 2. Selecting element cards</em></p>

<li>exporting data and downloading as XLSX file.</li>
<p>When the data was fully scraped and exported as a XLSX file, it had been edited in the Excel program. The datasets cleaning process and the results are shown in the figures 3 and 4 respectively. In the before figure, there were unnecessary columns, values, missing values, etc.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/0d8367d8-29ea-462b-96a9-84dad64940b2)

<p><em>Figure 3. Before data cleaning process in Excel</em></p>

<p>After the data cleaning process and feature engineering, new columns were added. Columns’ data types were changed.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/3ac48ac1-816a-406d-bda5-59ade8287a20)

<p><em>Figure 4. After data cleaning process in Excel</em></p>
</ol>
<p>In order to add needed features for data preprocessing, Google Sheets extention called “Geocode by Awesome Table” was used. It geocoded all the addresses with their longitude and latitude.</p>


<h2>Data analysis</h2>

<p>Data analysis is a set of methods and tools for extracting information from organized data for decision making. Analysis is not just processing information after it has been received and collected, it is a tool for testing hypotheses. The goal of any data analysis is to fully understand the situation under study (identify trends, including negative deviations from the plan, make predictions and make recommendations). To achieve this goal, the following data analysis tasks are set:</p>
<un>
<li>gathering information;</li>
<li>information structuring;</li>
<li>identification and analysis of laws;</li>
<li>predict and receive offers.</li>
</un>

<p>All data contains important information, but for different questions. All arrays need to be processed to extract useful data for specific situations.</p>
 
<p>In the process of data processing, preparation for analysis was carried out, as a result of which they were adjusted to the requirements determined by the specifics of the problem to be solved.</p>
<p>Pre-processing is an important stage of Data Mining, and if it is not performed, in the subsequent analysis, in many cases, analytical algorithms may be hindered or the results of their work may be incorrect. In other words, the principle of GIGO - garbage in, garbage out (garbage at the entrance, garbage at the exit) is implemented.</p>
<p>Data processing involves cleaning and optimizing data to prevent factors that diminish data quality and impede analytical algorithms, addressing duplicates, conflicts, false positives, missing values, noise reduction, outlier editing, and restoring the structure, completeness, integrity, and correct formatting of the data during the cleaning process. Data preprocessing and cleaning are important tasks that must be performed before using a dataset to train a model.</p>
<p>Raw data are often skewed and unreliable, and may contain missing values.</p>
<p>Using such data in modeling can lead to incorrect results.</p>
<p>Real data is collected for processing after various sources and processes. They may contain errors and damages that negatively affect the quality of the data set. These may be typical data quality problems:</p>
<un>
<li>incomplete: the data has no attributes or no values;</li>
<li>noise: the data contains incorrect entries or outliers;</li>
<li>discrepancy: data consists of conflicting records or gaps.</li>
</un>
<p>The general view of the data after the data cleaning process is shown in figure 5. There are 6700 rows of data.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/9d5fa03d-4a1b-4b58-9943-452a116c5e52)

<p><em>Figure 5. General view of the dataset</em></p>

<p>The next step is to create a preprocessing process for this collected data. The process started by loading the dataset in Jupyter Notebook. Information in the figure 6 indicates the number and types of available parameters.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/57f5ee8d-a6a8-402d-b0b7-02d385721b71)

<p><em>Figure 6. Information about the data frame</em></p>

<p>Wallmaterial is a column with the type of wall materials. Values for each type are shown in the figure 7.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/decfbede-7a7a-45c8-a164-7a16a79db5d5)

<p><em>Figure 7. Information about wallmaterial</em></p>

<p>Values that had NaN type were removed and refilled, as in the figure 8. NaN, standing for Not a Number, is a particular value of a numeric data type which is undefined or unrepresentable, such as the result of zero divided by zero.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/25898f2c-59d1-41b3-af14-d010fb91d92d)

<p><em>Figure 8. Filling missing values in wallmaterial</em></p>

<p>floorNumber is the number of the floor the apartment is located at. floorTotal is the number of total floors in the building. totalArea is the total area of the apartment. Missing values in these columns were less than 1%, therefore it was filled with average values. state is the state of the apartment. The types of states are shown in the figure 9.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/87244c4a-c77f-46fa-b2db-61c208115dc2)

<p><em>Figure 9. Information about state</em></p>

<p>Coordinates of the apartments - longitude and latitude data - are shown in the figure 10.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/d5c24d50-6b95-461d-8e6a-4f2e4ed245ef)

<p><em>Figure 10. Information about longitude and latitude</em></p>

<p>Price column’s descriptive statistics is shown in the figure 11. Mean price of all apartments is about 50 million KZT. It also shows that there is an apartment that is listed for 1 billion KZT.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/07d1268a-b19c-4371-ba5c-2c3102493822)

<p><em>Figure 11. Information about price</em></p>

<p>Year column’s descriptive statistics is shown in the figure 12. The oldest building listed was built in 1932. 50% of buildings were built in 2011.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/ea9a4d10-545d-41bf-9db2-8b4808bb0e5f)

<p><em>Figure 12. Information about year</em></p>

<p>Feature engineering is the process of selecting, transforming, and creating features or variables from raw data in order to improve the performance of a machine learning model. This involves identifying the relevant variables, removing irrelevant or redundant ones, transforming variables to improve their quality, and creating new variables that may be useful for the model. Feature engineering is a crucial step in the data analysis process as it can significantly impact the accuracy and generalizability of a model. It requires a deep understanding of the data and the problem domain, as well as creativity and knowledge of various techniques and tools for feature selection and transformation.</p>
<p>Following features were added:</p>
<ul>
<li>priceMetr is the area of the apartment per square meter;</li>
<li>distance is the distance from the apartment to the city center;</li>
<li>azimuth is the angle relative to the north direction.</li>
</ul>
<p>As can be seen in the figure 13, there were two categorical parameters after filling in incomplete data.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/321fdb18-6796-40d3-8add-a2b98fa4ec96)

<p><em>Figure 13. Parameters with categorical values</em></p>

<p>To work with these parameters, they were converted to numeric values, as shown in the figure 14. 3 values were assigned to “Wallmaterial” column and 5 values were assigned to “State” column.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/034b9e7d-2dca-438c-81be-933b6a30b1ed)

<p><em>Figure 14. Converting categorical parameters to numeric</em></p>

<p>As a result, the general view of the data is shown in the figure 15. It has 12 columns, meaning a dataset has 12 features.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/763a0b05-c4a5-4d8f-b646-25a7b807558a)

<p><em>Figure 15. Data after pre-processing</em></p>

<p>The last stage of data preparation is to select the target variable that for prediction. It is the price of an apartment per square meter. The next step is to select the parameters that will take part in the training of the model and form a new X dataset with no price indication, as in the figure 16.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/3aa5a4ab-3e8d-4372-9034-9c2b2408b12b)

<p><em>Figure 16. Machine learning parameters</em></p>

<p>The train_test_split() function automatically divides X and y into 4 groups. This allows to check the quality of the model for unfamiliar data.</p>
<un>
<li>train set of X;</li>
<li>test set of X;</li>
<li>train set of y;</li>
<li>test set of y.</li>
</un>

<h2>Model training and evaluation</h2>

<p>Model training and testing is a critical step in the development of a machine learning model. During training, the model is exposed to a set of labeled data, which it uses to learn the patterns and relationships between input features and output labels. The goal is to minimize the difference between the predicted output and the true output for each input. Once training is complete, the model is evaluated on a separate set of data, known as the test set, to assess its performance and generalizability. The results of the evaluation are then used to make any necessary adjustments to the model or to choose a different model altogether.</p>
<p>For the real estate price prediction task, following machine learning algorithms were used:</p>
<un>
<li>Random Forest Regressor;</li>
<li>XGB Regressor;</li>
<li>Linear Regression;</li>
<li>Decision Tree Regressor;</li>
<li>Ridge;</li>
<li>Lasso;</li>
<li>Keras deep learning model.</li></un>
<p>Development and training of a neural network model using Keras library has several stages. The necessary modules from TensorFlow and Keras are imported. The custom loss function called root_mean_squared_error_keras was defined. This function calculates the root mean squared error between the true labels (y_true) and predicted labels (y_pred). A sequential model is created using the Sequential class from Keras. This model consists of multiple layers defined using the Dense class. Each Dense layer represents a fully connected layer in the neural network. The specified number of units in each layer determines the dimensionality of the output space. The activation function relu (Rectified Linear Unit) is used for the intermediate layers, while the final layer is left without an activation function. The model is compiled by specifying the optimizer and loss function. In this case, the RMSprop optimizer is used, and the custom loss function root_mean_squared_error_keras is utilized. The fit function is called to train the model. It takes the training data train_X and corresponding labels train_y as input. The validation_data parameter is used to evaluate the model's performance on the validation data during training. The batch_size specifies the number of samples used in each gradient update, and epochs determine the number of times the training data will be iterated over.</p>
<p>The metrics for evaluating the models are:</p>
<un>
<li>mean absolute error;</li>
<li>median absolute error;</li>
<li>mean squared error;</li>
<li>root mean squared error.</li>
</un>
<p>Mean absolute error measures the average absolute difference between the predicted values and the true values. It provides a measure of the average magnitude of the errors.</p>
<p>Median absolute error is similar to MAE, but instead of taking the mean of the absolute differences, it takes the median. The median represents the middle value in a sorted list of values. MedAE is less sensitive to outliers compared to MAE and provides a robust measure of error.</p>
<p>Mean squared error measures the average of the squared differences between the predicted values and the true values. It gives higher weight to large errors since errors are squared before averaging.</p>
<p>Root mean squared error penalizes large errors more compared to MAE since it involves the square of the differences. It provides a measure of the typical magnitude of errors and is sensitive to outliers.</p>
<p>The trained models’ performances and scores are shown in the figure 17.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/2c9e84d3-4133-49e9-9bd3-03b32b80681e)

<p><em>Figure 17. Models’ results</em></p>
<p>Based on the results, the random forest was chosen as the main model for the web application.</p>
<p>To train the random forest algorithm, a RandomForestRegressor object was created. It was saved as rf_model. And it includes a number of parameters (i.e. hyperparameters), namely:</p>
<un>
<li>number of trees: 2000 (n_estimators=2000);</li>
<li>maximum tree depth: 55 (max_depth=55);</li>
<li>number of tasks running in parallel: (n_jobs=-1);</li>
<li>quality measurement function: (criterion= 'mse');</li>
<li>number	of	functions	to	consider	when	searching	for	a	division: (max_features=3);</li>
<li>minimum number of samples: (min_samples_split=2).</li>
</un>
<p>Launching the model was done by using the fit method. The results of a random forest model is shown in the figure 18.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/940ecc8f-335b-4859-b47c-7da81ec084a2)

<p><em>Figure 18. Results of random forest model</em></p>

<p>The predict method (val_X) validates the value of the X dataset. And the print_metrics() function takes predetermined and and predetermined estimates and prints metric values.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/88d9ae5c-8ab4-43ee-a269-d7895885a13c)
 
<p><em>Figure 19. Important features ranking</em></p>

<p>In creating a model, there is a way to see the importance of labels when creating a random forest. In the figure 19, ranking of important features are shown in the following way:</p>
<ol>
<li>distance (0.339830);</li>
<li>azimuth (0.186889);</li>
<li>totalArea (0.179689);</li>
<li>year (0.107004);</li>
<li>floorsTotal (0.082462);</li>
<li>floorNumber (0.054648); </li>
<li>state (0.027602);</li>
<li>wallmaterial (0.021874).</li>
</ol>

<h2>Model deployment</h2>

<p>To check the performance of the model after training, the price of an apartment put up for sale on the website https://krisha.kz was predicted. The features of the apartment are shown in the figure 20.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/b7038e0b-98b6-4437-bda7-5f3d38cae187)

<p><em>Figure 21. Parameters of an apartment</em></p>
 
<p>To do this, a dataframe describing the parameters of this apartment was made, as shown in the figure 22. 10 features of an apartment were inputted.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/82c585a4-2d98-428c-b527-54976773adfc)

<p><em>Figure 22. Creating dataframe</em></p>

<p>Excess elements were removed from the DataFrame by filling in the missing parameters with the available parameters. The drop function was used for this. The offer price was predicted through the advanced rf_model.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/8adcc85b-b1d1-421f-b61a-82095d226ba8)

<p><em>Figure 23. Prediction result</em></p>

<p>The price of the apartment predicted by the model shown in the figure 23 – 33881000 tenge. As an 8-10% error was expected, the apartment's real price was lower than the predicted price of 2.6%.</p>

<h2>Real estate price prediction web application development</h2>

<p>Several machine learning models were developed, made numerical predictions for this test, verified the results, and did it all autonomously. Actually, the generation of predictions is only one part of the machine learning project, but at the same time, it was the most important part.</p>
 
<p>A DFD (Data Flow Diagram) is a graphical representation that illustrates the flow of data within a system. It shows how data is input, processed, and outputted by different components of the system. The DFD diagram is shown in the figure 4.6.1.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/c8d70621-8239-485e-a68d-c2546e955d72)

<p><em>Figure 4.6.1. Model as a service for users</em></p>

<p>The web application is built on the basis of a model built on machine learning. After training the model, the model was saved so that it could be used without retraining. The following lines were added to save the model as a file .PKL for further use of the file. To load the model in the web app, the model load function was used.</p>
<p>The application was launched as a single module. To do this, a new Flask instance was started with the name argument to see if it can find the Flask template in the same directory as HTML (templates). Next, route decorator ( @app) was used, which activates the index function to display the URL. The Post method was used to transfer data to the server. Within the predict function, it accesses a dataset to predict several parameters of the apartment by taking it through the form [18]. The model takes the new values entered by the user and use it to predict the price of the apartment. To predict the price of an apartment, user needs to fill out a form for predicting several parameters of the apartment and click the predict button.</p>
<p>In the figure 4.6.2, the front and main page of the web application is shown. The website’s IP is on the address 127.0.0.1:5000. It has a menu bar and a landing page.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/885b59f5-c0f4-4963-bb29-d8f4104fc8da)

<p><em>Figure 4.6.2. Main page</em></p>

<p>The form for inputting apartment details is shown in the figure 4.6.3. It allows to input 10 features of an apartment.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/e26418a5-d689-4d7d-9a99-e8525e2118a8)

<p><em>Figure 4.6.3. Apartment details inputting form</em></p>

<p>The apartment details for price prediction is shown in the figure 4.6.4. It is listed in the krisha.kz. 2 bedroom apartment, with 46.5 sqrm is on the list for 29 million KZT. It is located on the 3rd floor of the 4-floor building. The building was built in 1068 and it has a panel wallmaterial.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/04ebed93-d9d5-48a2-888b-6d85730c8833)

<p><em>Figure 4.6.4. Apartment details for price prediction</em></p>
 
<p>The prediction process is shown in the figures 4.6.5 and 4.6.6 respectively. After inputting the the details, it is necessary to click “Submit” button.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/2f0d9e5a-f4fa-4d60-87a3-80e7f90d8cf0)

<p><em>Figure 4.6.5. Apartment details inputting for price prediction</em></p>

<p>Within a second, the machine learning model takes apartment details as an input and calculates the price of an apartment.</p>

![image](https://github.com/okaigerim/graduationproject/assets/64651950/3a8f0286-e050-41a3-bce6-885b3205268a)

<p><em>Figure 4.6.6. Apartment price prediction</em></p>

<p>Initial price of an apartment was 29000000 KZT, whereas the model gave and output of 28600000 KZT [20]. As expected from the machine learning model, it showed an error withing 8-12% range.</p>
