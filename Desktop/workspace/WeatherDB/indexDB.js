require('dotenv').config();
const fs = require('fs');
const path = require('path');
const mysql = require('mysql2');

// Load environment variables
const { DB_HOST, DB_USER, DB_PASSWORD, DB_NAME } = process.env;

// Configure the MySQL connection with multipleStatements enabled
const connection = mysql.createConnection({
    host: DB_HOST,
    user: DB_USER,
    password: DB_PASSWORD,
    database: DB_NAME, // Connect directly to the target database
    multipleStatements: true // Enable multiple statements
});

// Connect to the MySQL server
connection.connect(err => {
    if (err) {
        console.error('Error connecting to the database:', err);
        return;
    }
    console.log('Connected to the database server.');

    // Read the SQL file
    const sqlFilePath = path.join(__dirname, 'db/sql/index_WeatherDB.sql');
    fs.readFile(sqlFilePath, 'utf8', (err, sql) => {
        if (err) {
            console.error(`Error reading SQL file: ${sqlFilePath}`, err);
            connection.end();
            return;
        }

        console.log(`Executing SQL file: ${sqlFilePath}`);
        // Execute the SQL commands to add indexes
        connection.query(sql, err => {
            if (err) {
                console.error('Error executing SQL file:', err.sqlMessage);
                console.log(err.sql);
                connection.end();
                return;
            }
            console.log('Indexes added successfully.');
            // Close the database connection after indexes are added
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