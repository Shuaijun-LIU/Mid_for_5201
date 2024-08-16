const connectDB = require('../config/connectDB');


// 获取所有设备
exports.getAllDevices = async (req, res) => {
    try {
        const [devices] = await connectDB.query('SELECT * FROM ObservationDevice');
        res.render('devices', { devices });
    } catch (error) {
        console.error(error);
        res.status(500).send('Server Error');
    }
};

// 添加新设备
exports.addDevice = async (req, res) => {
    const { device_name, model, serial_number, status } = req.body;

    try {
        await connectDB.query(
            'INSERT INTO ObservationDevice (device_name, model, serial_number, status) VALUES (?, ?, ?, ?)',
            [device_name, model, serial_number, status]
        );
        res.redirect('/devices');
    } catch (error) {
        console.error(error);
        res.status(500).send('Server Error');
    }
};

// 获取单个设备信息
exports.getDeviceById = async (req, res) => {
    const { id } = req.params;

    try {
        const [device] = await connectDB.query('SELECT * FROM ObservationDevice WHERE device_id = ?', [id]);
        if (device.length === 0) {
            return res.status(404).send('Device not found');
        }
        res.render('editDevice', { device: device[0] });
    } catch (error) {
        console.error(error);
        res.status(500).send('Server Error');
    }
};

// 更新设备信息
exports.updateDevice = async (req, res) => {
    const { id } = req.params;
    const { device_name, model, serial_number, status } = req.body;

    try {
        await connectDB.query(
            'UPDATE ObservationDevice SET device_name = ?, model = ?, serial_number = ?, status = ? WHERE device_id = ?',
            [device_name, model, serial_number, status, id]
        );
        res.redirect('/devices');
    } catch (error) {
        console.error(error);
        res.status(500).send('Server Error');
    }
};

// 删除设备
exports.deleteDevice = async (req, res) => {
    const { id } = req.params;

    try {
        await connectDB.query('DELETE FROM ObservationDevice WHERE device_id = ?', [id]);
        res.redirect('/devices');
    } catch (error) {
        console.error(error);
        res.status(500).send('Server Error');
    }
};