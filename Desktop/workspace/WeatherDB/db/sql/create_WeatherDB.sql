-- Create dimension tables
CREATE TABLE BloodType (
    blood_type_id INT PRIMARY KEY,
    blood_type VARCHAR(3)
);

CREATE TABLE Country (
    country_id CHAR(2) PRIMARY KEY,
    country_name VARCHAR(50)
);

CREATE TABLE Ethnicity (
    ethnicity_type_id INT PRIMARY KEY,
    ethnicity_type VARCHAR(50)
);

CREATE TABLE Department (
    department_id INT PRIMARY KEY,
    department_name VARCHAR(100),
    creation_date DATE,
    department_description TEXT
);

CREATE TABLE Employee (
    employee_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    dob DATE,
    phone VARCHAR(15),
    email VARCHAR(100),
    sex VARCHAR(10),
    marital_status VARCHAR(10),
    ethnicity_type_id INT,
    country_id CHAR(2),
    blood_type_id INT,
    department_id INT,
    FOREIGN KEY (ethnicity_type_id) REFERENCES Ethnicity(ethnicity_type_id),
    FOREIGN KEY (country_id) REFERENCES Country(country_id),
    FOREIGN KEY (blood_type_id) REFERENCES BloodType(blood_type_id),
    FOREIGN KEY (department_id) REFERENCES Department(department_id)
);

CREATE TABLE ObservationDevice (
    device_id INT PRIMARY KEY,
    device_name VARCHAR(100),
    model VARCHAR(50),
    serial_number VARCHAR(50),
    status VARCHAR(20),
    installation_date DATE,
    sensor_accuracy VARCHAR(10),
    maintainer_id INT,
    FOREIGN KEY (maintainer_id) REFERENCES Employee(employee_id)
);

CREATE TABLE State (
    state_id CHAR(2) PRIMARY KEY,
    state_name VARCHAR(50)
);

CREATE TABLE Station (
    station_id BIGINT PRIMARY KEY,
    station_name VARCHAR(100),
    begin_date DATE,
    end_date DATE,
    state_id CHAR(2),
    country_id CHAR(2),
    latitude FLOAT,
    longitude FLOAT,
    elevation FLOAT,
    FOREIGN KEY (state_id) REFERENCES State(state_id),
    FOREIGN KEY (country_id) REFERENCES Country(country_id)
);

CREATE TABLE Time (
    time_id INT PRIMARY KEY,
    Year INT,
    Month INT,
    Day INT,
    Hour INT,
    quarter VARCHAR(2),
    day_session VARCHAR(10)
);

CREATE TABLE EventStatus (
    indicator_status_type INT PRIMARY KEY,
    indicator_status_name VARCHAR(20)
);

-- Create fact tables
CREATE TABLE Observation (
    time_id INT,
    station_id BIGINT,
    Temperature FLOAT,
    DewPointTemperature FLOAT,
    Pressure FLOAT,
    WindDirection INT,
    WindSpeed FLOAT,
    CloudCover FLOAT,
    Rain1h FLOAT,
    Rain6h FLOAT,
    observer_id INT,
    observation_device_id INT,
    PRIMARY KEY (time_id, station_id, observer_id, observation_device_id),
    FOREIGN KEY (time_id) REFERENCES Time(time_id),
    FOREIGN KEY (station_id) REFERENCES Station(station_id),
    FOREIGN KEY (observer_id) REFERENCES Employee(employee_id),
    FOREIGN KEY (observation_device_id) REFERENCES ObservationDevice(device_id)
);

CREATE TABLE SpecialEvent (
    time_id INT,
    station_id BIGINT,
    Temperature_status INT,
    DewPointTemperature_status INT,
    Pressure_status INT,
    WindDirection_status INT,
    WindSpeed_status INT,
    CloudCover_status INT,
    Rain1h_status INT,
    Rain6h_status INT,
    event_id INT,
    observer_id INT,
    observation_device_id INT,
    PRIMARY KEY (time_id, station_id, event_id, observer_id, observation_device_id),
    FOREIGN KEY (time_id) REFERENCES Time(time_id),
    FOREIGN KEY (station_id) REFERENCES Station(station_id),
    FOREIGN KEY (observer_id) REFERENCES Employee(employee_id),
    FOREIGN KEY (observation_device_id) REFERENCES ObservationDevice(device_id),
    FOREIGN KEY (Temperature_status) REFERENCES EventStatus(indicator_status_type),
    FOREIGN KEY (DewPointTemperature_status) REFERENCES EventStatus(indicator_status_type),
    FOREIGN KEY (Pressure_status) REFERENCES EventStatus(indicator_status_type),
    FOREIGN KEY (WindDirection_status) REFERENCES EventStatus(indicator_status_type),
    FOREIGN KEY (WindSpeed_status) REFERENCES EventStatus(indicator_status_type),
    FOREIGN KEY (CloudCover_status) REFERENCES EventStatus(indicator_status_type),
    FOREIGN KEY (Rain1h_status) REFERENCES EventStatus(indicator_status_type),
    FOREIGN KEY (Rain6h_status) REFERENCES EventStatus(indicator_status_type)
);