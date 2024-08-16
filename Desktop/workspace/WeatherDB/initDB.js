require('dotenv').config();
const fs = require('fs');
const path = require('path');
const mysql = require('mysql2');
const JSONStream = require('JSONStream');

// Load environment variables
const { DB_HOST, DB_USER, DB_PASSWORD, DB_NAME } = process.env;

// Configure the MySQL connection with multipleStatements enabled
const connection = mysql.createConnection({
    host: DB_HOST,
    user: DB_USER,
    password: DB_PASSWORD,
    multipleStatements: true // Enable multiple statements
});

// Connect to the MySQL server
connection.connect(err => {
    if (err) {
        console.error('Error connecting to the database:', err);
        return;
    }
    console.log('Connected to the database server.');

    // Check if the database exists
    connection.query(`SHOW DATABASES LIKE '${DB_NAME}'`, (err, results) => {
        if (err) {
            console.error('Error checking for database existence:', err);
            connection.end();
            return;
        }

        if (results.length > 0) {
            // Drop the database if it exists
            console.log(`Database ${DB_NAME} exists. Dropping database...`);
            connection.query(`DROP DATABASE ${DB_NAME}`, err => {
                if (err) {
                    console.error('Error dropping database:', err);
                    connection.end();
                    return;
                }
                console.log(`Database ${DB_NAME} dropped successfully.`);
                createDatabase();
            });
        } else {
            createDatabase();
        }
    });
});

// Function to create the database and import data
function createDatabase() {
    // Create a new database
    connection.query(`CREATE DATABASE ${DB_NAME}`, err => {
        if (err) {
            console.error('Error creating database:', err);
            connection.end();
            return;
        }
        console.log(`Database ${DB_NAME} created successfully.`);

        // Switch to the new database
        connection.changeUser({ database: DB_NAME }, err => {
            if (err) {
                console.error('Error switching to the new database:', err);
                connection.end();
                return;
            }

            // Read and execute SQL file to create tables
            const sqlFilePath = path.join(__dirname, 'db/sql/create_WeatherDB.sql');
            fs.readFile(sqlFilePath, 'utf8', (err, sql) => {
                if (err) {
                    console.error(`Error reading SQL file: ${sqlFilePath}`, err);
                    connection.end();
                    return;
                }

                console.log(`Executing SQL file: ${sqlFilePath}`);
                connection.query(sql, err => {
                    if (err) {
                        console.error('Error executing SQL file:', err.sqlMessage);
                        console.log(err.sql);
                        connection.end();
                        return;
                    }
                    console.log('Database schema created successfully.');

                    // Now proceed with data import
                    importData(() => {
                        // Close the database connection after data import is complete
                        connection.end(err => {
                            if (err) {
                                console.error('Error closing the database connection:', err);
                            } else {
                                console.log('Database connection closed.');
                            }
                        });
                    });
                });
            });
        });
    });
}

// Function to import data from JSON files using streaming
function importData(callback) {
    // Set the flag file path
    const flagFilePath = path.join(__dirname, 'import_flag.txt');

    // Check if the flag file exists
    if (fs.existsSync(flagFilePath)) {
        console.log('Data has already been imported.');
        callback();
        return;
    }

    console.log('Starting import process...');

    // Define the directory where the JSON files are located
    const jsonDirectory = path.join(__dirname, 'db/dataset/');

    // Function to import data using streaming
    function importSingleFile(jsonFile, tableName, columns) {
        return new Promise((resolve, reject) => {
            const jsonPath = path.join(jsonDirectory, jsonFile);
            if (!fs.existsSync(jsonPath)) {
                return reject(new Error(`Error: ${jsonPath} does not exist.`));
            }

            console.log(`Importing data from ${jsonFile} into table ${tableName}`);
            const stream = fs.createReadStream(jsonPath, { encoding: 'utf8' });
            const parser = JSONStream.parse('*');

            let bulkInsert = [];
            let rowCount = 0;

            stream.pipe(parser);

            parser.on('data', (item) => {
                try {
                    const values = columns.map(column => {
                        let value = item[column];
                        if (value === undefined || value === null) {
                            value = ''; // Handle undefined or null values
                        } else {
                            value = value.toString().replace(/ /g, '_').trim();
                        }
                        return connection.escape(value);
                    });

                    bulkInsert.push(`(${values.join(',')})`);
                    rowCount++;

                    if (bulkInsert.length >= 1000) { // Batch size of 1000
                        const sql = `INSERT INTO ${tableName} (${columns.join(',')}) VALUES ${bulkInsert.join(',')}`;
                        connection.query(sql, (err) => {
                            if (err) {
                                console.error(`Error inserting batch at row ${rowCount} into table ${tableName}:`, err);
                                reject(err);
                            }
                        });
                        console.log(`Imported batch of 1000 rows into ${tableName}`);
                        bulkInsert = [];
                    }
                } catch (error) {
                    console.error(`Error processing data at row ${rowCount} for ${tableName}:`, item);
                    reject(error);
                }
            });

            parser.on('end', () => {
                if (bulkInsert.length > 0) {
                    const sql = `INSERT INTO ${tableName} (${columns.join(',')}) VALUES ${bulkInsert.join(',')}`;
                    connection.query(sql, (err) => {
                        if (err) {
                            console.error(`Error inserting final batch into table ${tableName}:`, err);
                            reject(err);
                        } else {
                            console.log(`Imported remaining ${bulkInsert.length} rows into ${tableName}`);
                            resolve();
                        }
                    });
                } else {
                    resolve();
                }
            });

            parser.on('error', (err) => {
                console.error(`Error parsing JSON file ${jsonFile}:`, err);
                reject(err);
            });

            stream.on('error', (err) => {
                console.error(`Error reading JSON file ${jsonFile}:`, err);
                reject(err);
            });
        });
    }

    // Chain the import of each JSON file
    importSingleFile('BloodType.json', 'BloodType', ['blood_type_id', 'blood_type'])
        .then(() => importSingleFile('Country.json', 'Country', ['country_id', 'country_name']))
        .then(() => importSingleFile('Ethnicity.json', 'Ethnicity', ['ethnicity_type_id', 'ethnicity_type']))
        .then(() => importSingleFile('Department.json', 'Department', ['department_id', 'department_name', 'creation_date', 'department_description']))
        .then(() => importSingleFile('Employee.json', 'Employee', ['employee_id', 'first_name', 'last_name', 'dob', 'phone', 'email', 'sex', 'marital_status', 'ethnicity_type_id', 'country_id', 'blood_type_id', 'department_id']))
        .then(() => importSingleFile('ObservationDevice.json', 'ObservationDevice', ['device_id', 'device_name', 'model', 'serial_number', 'status', 'installation_date', 'sensor_accuracy', 'maintainer_id']))
        .then(() => importSingleFile('State.json', 'State', ['state_id', 'state_name']))
        .then(() => importSingleFile('Station.json', 'Station', ['station_id', 'station_name', 'begin_date', 'end_date', 'state_id', 'country_id', 'latitude', 'longitude', 'elevation']))
        .then(() => importSingleFile('Time.json', 'Time', ['time_id', 'Year', 'Month', 'Day', 'Hour', 'quarter', 'day_session']))
        .then(() => importSingleFile('EventStatus.json', 'EventStatus', ['indicator_status_type', 'indicator_status_name']))
        .then(() => importSingleFile('Observation.json', 'Observation', [
            'time_id', 'station_id', 'Temperature', 'DewPointTemperature', 'Pressure',
            'WindDirection', 'WindSpeed', 'CloudCover', 'Rain1h', 'Rain6h', 
            'observer_id', 'observation_device_id'
        ]))
        .then(() => importSingleFile('SpecialEvent.json', 'SpecialEvent', [
            'time_id', 'station_id', 'Temperature_status', 'DewPointTemperature_status',
            'Pressure_status', 'WindDirection_status', 'WindSpeed_status',
            'CloudCover_status', 'Rain1h_status', 'Rain6h_status', 'event_id',
            'observer_id', 'observation_device_id'
        ]))
        .then(() => {
            // Create the flag file after import
            fs.writeFile(flagFilePath, 'Data imported successfully.', err => {
                if (err) {
                    console.error(`Error creating flag file: ${flagFilePath}`, err.message);
                    process.exit();
                }
                console.log('All data has been imported successfully.');
                callback();
            });
        })
        .catch(err => {
            console.error('Error during import process:', err.message);
            connection.end();
        });
}