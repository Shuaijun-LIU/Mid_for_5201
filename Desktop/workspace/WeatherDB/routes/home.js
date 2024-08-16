const express = require('express');
const router = express.Router();
const homeController = require('../controllers/homeController');
const path = require('path');

// Serve the index.html file for the root path
router.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/home.html'));
});

router.get('/api/dashboard', homeController.getDashboardData);
router.post('/api/query', homeController.executeQuery);

module.exports = router;