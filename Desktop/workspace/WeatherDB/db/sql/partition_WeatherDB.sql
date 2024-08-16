-- Step 1: Drop foreign key constraints
-- Drop foreign keys from Observation table
ALTER TABLE Observation DROP FOREIGN KEY observation_ibfk_1;
ALTER TABLE Observation DROP FOREIGN KEY observation_ibfk_2;
ALTER TABLE Observation DROP FOREIGN KEY observation_ibfk_3;
ALTER TABLE Observation DROP FOREIGN KEY observation_ibfk_4;

-- Drop foreign keys from SpecialEvent table
ALTER TABLE SpecialEvent DROP FOREIGN KEY specialevent_ibfk_1;
ALTER TABLE SpecialEvent DROP FOREIGN KEY specialevent_ibfk_2;
ALTER TABLE SpecialEvent DROP FOREIGN KEY specialevent_ibfk_3;
ALTER TABLE SpecialEvent DROP FOREIGN KEY specialevent_ibfk_4;
ALTER TABLE SpecialEvent DROP FOREIGN KEY specialevent_ibfk_5;
ALTER TABLE SpecialEvent DROP FOREIGN KEY specialevent_ibfk_6;
ALTER TABLE SpecialEvent DROP FOREIGN KEY specialevent_ibfk_7;
ALTER TABLE SpecialEvent DROP FOREIGN KEY specialevent_ibfk_8;
ALTER TABLE SpecialEvent DROP FOREIGN KEY specialevent_ibfk_9;
ALTER TABLE SpecialEvent DROP FOREIGN KEY specialevent_ibfk_10;
ALTER TABLE SpecialEvent DROP FOREIGN KEY specialevent_ibfk_11;
ALTER TABLE SpecialEvent DROP FOREIGN KEY specialevent_ibfk_12;


-- Step 2: Partition the tables
-- Partition Observation table
ALTER TABLE Observation
PARTITION BY RANGE (time_id) (
    PARTITION p2020 VALUES LESS THAN (8777),  -- 2020
    PARTITION p2021 VALUES LESS THAN (17487), -- 2021
    PARTITION p2022 VALUES LESS THAN (26247), -- 2022
    PARTITION p2023 VALUES LESS THAN (35000), -- 2023
    PARTITION p2024 VALUES LESS THAN (40072), -- 2024
    PARTITION p_future VALUES LESS THAN MAXVALUE -- Future
);

-- Partition SpecialEvent table
ALTER TABLE SpecialEvent
PARTITION BY RANGE (time_id) (
    PARTITION p2020 VALUES LESS THAN (8777),  -- 2020
    PARTITION p2021 VALUES LESS THAN (17487), -- 2021
    PARTITION p2022 VALUES LESS THAN (26247), -- 2022
    PARTITION p2023 VALUES LESS THAN (35000), -- 2023
    PARTITION p2024 VALUES LESS THAN (40072), -- 2024
    PARTITION p_future VALUES LESS THAN MAXVALUE -- Future
);


-- Step 3: Re-add foreign key constraints
-- Re-add foreign keys to Observation table
ALTER TABLE Observation
ADD CONSTRAINT observation_ibfk_1 FOREIGN KEY (time_id) REFERENCES Time(time_id),
ADD CONSTRAINT observation_ibfk_2 FOREIGN KEY (station_id) REFERENCES Station(station_id),
ADD CONSTRAINT observation_ibfk_3 FOREIGN KEY (observer_id) REFERENCES Employee(employee_id),
ADD CONSTRAINT observation_ibfk_4 FOREIGN KEY (observation_device_id) REFERENCES ObservationDevice(device_id);

-- Re-add foreign keys to SpecialEvent table
ALTER TABLE SpecialEvent
ADD CONSTRAINT specialevent_ibfk_1 FOREIGN KEY (time_id) REFERENCES Time(time_id),
ADD CONSTRAINT specialevent_ibfk_2 FOREIGN KEY (station_id) REFERENCES Station(station_id),
ADD CONSTRAINT specialevent_ibfk_3 FOREIGN KEY (observer_id) REFERENCES Employee(employee_id),
ADD CONSTRAINT specialevent_ibfk_4 FOREIGN KEY (observation_device_id) REFERENCES ObservationDevice(device_id),
ADD CONSTRAINT specialevent_ibfk_5 FOREIGN KEY (Temperature_status) REFERENCES EventStatus(indicator_status_type),
ADD CONSTRAINT specialevent_ibfk_6 FOREIGN KEY (DewPointTemperature_status) REFERENCES EventStatus(indicator_status_type),
ADD CONSTRAINT specialevent_ibfk_7 FOREIGN KEY (Pressure_status) REFERENCES EventStatus(indicator_status_type),
ADD CONSTRAINT specialevent_ibfk_8 FOREIGN KEY (WindDirection_status) REFERENCES EventStatus(indicator_status_type),
ADD CONSTRAINT specialevent_ibfk_9 FOREIGN KEY (WindSpeed_status) REFERENCES EventStatus(indicator_status_type),
ADD CONSTRAINT specialevent_ibfk_10 FOREIGN KEY (CloudCover_status) REFERENCES EventStatus(indicator_status_type),
ADD CONSTRAINT specialevent_ibfk_11 FOREIGN KEY (Rain1h_status) REFERENCES EventStatus(indicator_status_type),
ADD CONSTRAINT specialevent_ibfk_12 FOREIGN KEY (Rain6h_status) REFERENCES EventStatus(indicator_status_type);