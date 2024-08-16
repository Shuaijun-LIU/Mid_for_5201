const express = require('express');
const router = express.Router();
const stationsController = require('../controllers/stationsController');

// 显示站点列表
router.get('/', stationsController.getAllStations);

// 添加新站点
router.post('/add', stationsController.addStation);

// 编辑站点信息
router.post('/edit/:id', stationsController.editStation);

// 删除站点
router.post('/delete/:id', stationsController.deleteStation);

module.exports = router;