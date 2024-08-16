const express = require('express');
const router = express.Router();
const devicesController = require('../controllers/devicesController');

// Serve the index.html file for the root path
router.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/devices.html'));
});


// 获取设备列表
router.get('/list', devicesController.getAllDevices);

// 获取单个设备
router.get('/:id', devicesController.getDeviceById);

// 添加新设备
router.post('/', devicesController.addDevice);

// 更新设备信息
router.put('/:id', devicesController.updateDevice);

// 删除设备
router.delete('/:id', devicesController.deleteDevice);

module.exports = router;