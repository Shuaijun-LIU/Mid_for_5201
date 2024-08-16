-- Add indexes to improve query performance
-- Indexes for Observation table
CREATE INDEX idx_observation_time_id ON Observation(time_id);
CREATE INDEX idx_observation_station_id ON Observation(station_id);
CREATE INDEX idx_observation_observer_id ON Observation(observer_id);
CREATE INDEX idx_observation_device_id ON Observation(observation_device_id);

-- Indexes for SpecialEvent table
CREATE INDEX idx_special_event_time_id ON SpecialEvent(time_id);
CREATE INDEX idx_special_event_station_id ON SpecialEvent(station_id);
CREATE INDEX idx_special_event_observer_id ON SpecialEvent(observer_id);
CREATE INDEX idx_special_event_device_id ON SpecialEvent(observation_device_id);
CREATE INDEX idx_special_event_temperature_status ON SpecialEvent(Temperature_status);
CREATE INDEX idx_special_event_dewpoint_status ON SpecialEvent(DewPointTemperature_status);
CREATE INDEX idx_special_event_pressure_status ON SpecialEvent(Pressure_status);
CREATE INDEX idx_special_event_winddirection_status ON SpecialEvent(WindDirection_status);
CREATE INDEX idx_special_event_windspeed_status ON SpecialEvent(WindSpeed_status);
CREATE INDEX idx_special_event_cloudcover_status ON SpecialEvent(CloudCover_status);
CREATE INDEX idx_special_event_rain1h_status ON SpecialEvent(Rain1h_status);
CREATE INDEX idx_special_event_rain6h_status ON SpecialEvent(Rain6h_status);

-- Indexes for Country table
CREATE INDEX idx_country_name ON Country(country_name);

-- Indexes for Department table
CREATE INDEX idx_department_name ON Department(department_name);

-- Indexes for Employee table
CREATE INDEX idx_employee_last_name ON Employee(last_name);
CREATE INDEX idx_employee_country_id ON Employee(country_id);
CREATE INDEX idx_employee_department_id ON Employee(department_id);

-- Indexes for ObservationDevice table
CREATE INDEX idx_observation_device_device_name ON ObservationDevice(device_name);
CREATE INDEX idx_observation_device_status ON ObservationDevice(status);
CREATE INDEX idx_observation_device_maintainer_id ON ObservationDevice(maintainer_id);

-- Indexes for Station table
CREATE INDEX idx_station_state_id ON Station(state_id);
CREATE INDEX idx_station_country_id ON Station(country_id);
CREATE INDEX idx_station_latitude ON Station(latitude);
CREATE INDEX idx_station_longitude ON Station(longitude);

-- Indexes for Time table
CREATE INDEX idx_time_year ON Time(Year);
CREATE INDEX idx_time_month ON Time(Month);
CREATE INDEX idx_time_day ON Time(Day);
CREATE INDEX idx_time_hour ON Time(Hour);
