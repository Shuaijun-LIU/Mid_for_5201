# WeatherDB Project

WeatherDB is a comprehensive data warehouse and web management system designed to manage and analyze meteorological data. It supports a range of functionalities for observing weather data, managing observation devices and stations, and handling department and employee information, providing an intuitive interface for meteorologists and weather station managers.

### 👍 Advisor
| **Prof. Jack Polnarh** |
| **Prof. Jim Gibson** |
|-------------------------|

### ✍️ Author
| **Shuaijun Liu** |
|-------------------------|

## <span style="color:orange;">🎨 Project Structure</span>

```
WeatherDB/
│
├── config/
│   └── connectDB.js
├── controllers/
│   ├── departmentsController.js
│   ├── devicesController.js
│   ├── employeesController.js
│   ├── homeController.js
│   ├── observationController.js
│   └── stationsController.js
├── db/
│   ├── dataset/
│   │   ├── 2020_2024_US/
│   │   │   └── stationid_2020_2024
│   │   ├── BloodType.json
│   │   ├── Country.json
│   │   ├── Department.json
│   │   ├── Employee.json
│   │   ├── Ethnicity.json
│   │   ├── Event.json
│   │   ├── EventStatus.json
│   │   ├── Observation.json
│   │   ├── ObservationDevice.json
│   │   ├── SpecialEvent.json
│   │   ├── State.json
│   │   └── Station.json
│   ├── description_files/
│   ├── ETL/
│   │   ├── us.csv
│   │   ├── 1.Extract_and_Transform.ipynb
│   │   ├── 2.ObserveData_Cleaning.ipynb
│   │   ├── 3.Special_event_data.ipynb
│   │   ├── 4.StationData_Processing.ipynb
│   │   └── 5.Find_time_id_for_Partition.ipynb
│   └── sql/
│       ├── create_WeatherDB.sql
│       ├── index_WeatherDB.sql
│       ├── partition_WeatherDB.sql
├── node_modules/
├── public/
│   ├── css/
│   │   ├── departments.css
│   │   ├── devices.css
│   │   ├── home.css
│   │   ├── observation.css
│   │   ├── stations.css
│   │   └── style.css
│   ├── images/
│   └── js/
├── routes/
├── .env
├── indexDB.js
├── initDB.js
├── server.js
├── package-lock.json
├── package.json
└── README.md
```

### <span style="color:lightblue;">📝 File Descriptions</span>

<ul>
  <li><strong><span style="color:mediumpurple;">config/</span></strong>: Contains the database connection settings.</li>
  <li><strong><span style="color:mediumpurple;">controllers/</span></strong>: Handles the business logic for different modules:
    <ul>
      <li><strong><span style="color:lightgreen;">departmentsController.js</span></strong>: Manages operations related to departments.</li>
      <li><strong><span style="color:lightgreen;">devicesController.js</span></strong>: Manages operations related to observation devices.</li>
      <li><strong><span style="color:lightgreen;">employeesController.js</span></strong>: Manages operations related to employees.</li>
      <li><strong><span style="color:lightgreen;">homeController.js</span></strong>: Handles the homepage logic.</li>
      <li><strong><span style="color:lightgreen;">observationController.js</span></strong>: Manages operations related to weather observations.</li>
      <li><strong><span style="color:lightgreen;">stationsController.js</span></strong>: Manages operations related to weather stations.</li>
    </ul>
  </li>
  <li><strong><span style="color:mediumpurple;">db/</span></strong>: Stores database-related files:
    <ul>
      <li><strong><span style="color:lightgreen;">dataset/</span></strong>: Contains raw data in JSON format and station ID files for 2020-2024.</li>
      <li><strong><span style="color:lightgreen;">ETL/</span></strong>: Jupyter notebooks and CSV files for data extraction, transformation, and loading.</li>
      <li><strong><span style="color:lightgreen;">sql/</span></strong>: SQL scripts for setting up the database, adding indexes, and partitioning tables.</li>
    </ul>
  </li>
  <li><strong><span style="color:mediumpurple;">node_modules/</span></strong>: Contains Node.js dependencies.</li>
  <li><strong><span style="color:mediumpurple;">public/</span></strong>: Contains static assets for the front-end, including CSS, images, and JavaScript files.
    <ul>
      <li><strong><span style="color:lightgreen;">css/</span></strong>: Stylesheets for different sections of the application.
        <ul>
          <li><strong><span style="color:lightcoral;">departments.css</span></strong>: Styles for department management pages.</li>
          <li><strong><span style="color:lightcoral;">devices.css</span></strong>: Styles for device management pages.</li>
          <li><strong><span style="color:lightcoral;">home.css</span></strong>: Styles for the home page.</li>
          <li><strong><span style="color:lightcoral;">observation.css</span></strong>: Styles for observation management pages.</li>
          <li><strong><span style="color:lightcoral;">stations.css</span></strong>: Styles for station management pages.</li>
          <li><strong><span style="color:lightcoral;">style.css</span></strong>: General styles for the application.</li>
        </ul>
      </li>
      <li><strong><span style="color:lightgreen;">images/</span></strong>: Image assets used in the application.</li>
      <li><strong><span style="color:lightgreen;">js/</span></strong>: JavaScript files for front-end interactions.</li>
    </ul>
  </li>
  <li><strong><span style="color:mediumpurple;">routes/</span></strong>: Expected to define the routing logic for the application.</li>
  <li><strong><span style="color:mediumpurple;">indexDB.js</span></strong>: Executes the `index_WeatherDB.sql` script to add indexes to the database.</li>
  <li><strong><span style="color:mediumpurple;">initDB.js</span></strong>: Initializes the WeatherDB database by running `create_WeatherDB.sql` and importing JSON data.</li>
  <li><strong><span style="color:mediumpurple;">server.js</span></strong>: Main server file that sets up the Node.js application and defines routes.</li>
  <li><strong><span style="color:mediumpurple;">package.json</span></strong>: Lists project dependencies and scripts.</li>
  <li><strong><span style="color:mediumpurple;">README.md</span></strong>: Project documentation file.</li>
</ul>

## <span style="color:orange;">🌐 Project Pages</span>

1. **Home Page**: Provides an overview of the system with charts displaying daily observations and device status.
2. **Observation Management**: Allows users to view, add, and edit weather observation records.
3. **Device Management**: Enables the management of observation devices, including adding new devices and updating existing ones.
4. **Station Management**: Handles station data, allowing for the addition and modification of station records.
5. **Department and Employee Management**: Manages department and employee information, including the ability to add and edit records.

## <span style="color:orange;">🚀 Getting Started</span>

### Prerequisites

Ensure you have the following installed:

- **Node.js** (v14.x or higher)
- **MySQL** (for database operations)

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/Shuaijun-LIU/WeatherDB
    cd weatherdb
    ```

2. **Install the dependencies:**
    ```sh
    npm install npm install express mysql2 body-parser dotenv
    ```

### Configuration

1. **Set up your MySQL database.** Create a database and configure the connection details in the `.env` file.

2. **Create a `.env` file in the project root and add the following configurations:**

    ```plaintext
    DB_HOST=localhost
    DB_USER=root
    DB_PASSWORD=yourpassword
    DB_NAME=weatherdb
    ```

### Running the Project
**Start the server using `node`:**
```sh
node server.js
```

### Accessing the Application

Once the server is running, open your web browser and navigate to:

```
http://localhost:3000
```

You will see the home page of the WeatherDB application.

## <span style="color:orange;">🌐 Accessing the Application</span>

Once the server is running, you can access the WeatherDB application by navigating to the following URL in your web browser:

```
http://localhost:3000
```

This will take you to the home page, where you can explore the various features of the system, including observation management, device management, station management, and department and employee management.

## <span style="color:orange;">🤝 Contributing</span>

We welcome contributions to the WeatherDB project! If you would like to contribute, please follow these steps:

1. **Fork the repository.**
2. **Create a new branch (`git checkout -b feature-branch`).**
3. **Make your changes.**
4. **Commit your changes (`git commit -am 'Add new feature'`).**
5. **Push to the branch (`git push origin feature-branch`).**
6. **Create a new Pull Request.**

Your contributions will be reviewed, and once approved, they will be merged into the main branch.

## <span style="color:orange;">📜 License</span>

This project is licensed under the **MIT License**. For more details, please refer to the `LICENSE` file in the repository.

## <span style="color:orange;">🙏 Acknowledgements</span>

- **National Oceanic and Atmospheric Administration (NOAA)** for providing the Integrated Surface Data (ISD) used in this project.
- **Boston University MET** for supporting this project.
- **All contributors** who have helped make WeatherDB a valuable tool for meteorologists and weather station managers.