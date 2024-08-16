const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
require('dotenv').config();

const homeRouter = require('./routes/home');
const observationRouter = require('./routes/observation');
const devicesRouter = require('./routes/devices');
// const stationsRouter = require('./routes/stations');
const departmentsRouter = require('./routes/departments');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware setup
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Set static folder
app.use(express.static(path.join(__dirname, 'public')));

// Route setup
app.use('/', homeRouter);
app.use('/observation', observationRouter);
app.use('/devices', devicesRouter);
// app.use('/stations', stationsRouter);
app.use('/departments', departmentsRouter);

// 404 Error handling
app.use((req, res, next) => {
    res.status(404).sendFile(path.join(__dirname, 'public', '404.html'));
});

// Global error handling
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something broke!');
});

// Start server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});