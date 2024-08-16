require('dotenv').config();
const mysql = require('mysql2');

const pool = mysql.createPool({
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
});

const promisePool = pool.promise();

promisePool.getConnection()
    .then(conn => {
        console.log('Database connection pool created successfully.');
        conn.release();
    })
    .catch(err => {
        console.error('Error creating database connection pool:', err);
    });

module.exports = promisePool;