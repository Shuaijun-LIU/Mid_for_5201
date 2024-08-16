const express = require('express');
const router = express.Router();
const observationController = require('../controllers/observationController');
const path = require('path');

// Serve the observation management page
router.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/observation.html'));
});

// API routes
router.get('/list', observationController.listObservations);

router.get('/:time_id/:station_id/:observation_device_id', observationController.getObservation);

router.post('/add', observationController.addObservation);

router.post('/edit', observationController.editObservation);

router.delete('/delete/:time_id/:station_id/:observation_device_id', observationController.deleteObservation);

module.exports = router;