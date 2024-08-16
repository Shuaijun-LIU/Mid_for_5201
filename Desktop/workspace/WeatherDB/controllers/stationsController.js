const db = require('../config/connectDB');

exports.listStations = async (req, res) => {
    try {
        const [rows] = await db.query('SELECT * FROM Station ORDER BY station_name ASC');
        res.json({ stations: rows });
    } catch (err) {
        console.error('Error fetching stations:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.getStation = async (req, res) => {
    const id = req.params.id;
    try {
        const [rows] = await db.query('SELECT * FROM Station WHERE station_id = ?', [id]);
        if (rows.length > 0) {
            res.json({ station: rows[0] });
        } else {
            res.status(404).json({ error: 'Station not found' });
        }
    } catch (err) {
        console.error('Error fetching station:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.addStation = async (req, res) => {
    const { station_name, begin_date, end_date, state_id, country_id, latitude, longitude, elevation } = req.body;
    try {
        const [result] = await db.query(
            'INSERT INTO Station (station_name, begin_date, end_date, state_id, country_id, latitude, longitude, elevation) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            [station_name, begin_date, end_date, state_id, country_id, latitude, longitude, elevation]
        );
        if (result.affectedRows === 1) {
            res.json({ success: true });
        } else {
            res.status(400).json({ error: 'Failed to add station' });
        }
    } catch (err) {
        console.error('Error adding station:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.editStation = async (req, res) => {
    const { station_id, station_name, begin_date, end_date, state_id, country_id, latitude, longitude, elevation } = req.body;
    try {
        const [result] = await db.query(
            'UPDATE Station SET station_name = ?, begin_date = ?, end_date = ?, state_id = ?, country_id = ?, latitude = ?, longitude = ?, elevation = ? WHERE station_id = ?',
            [station_name, begin_date, end_date, state_id, country_id, latitude, longitude, elevation, station_id]
        );
        if (result.affectedRows === 1) {
            res.json({ success: true });
        } else {
            res.status(400).json({ error: 'Failed to update station' });
        }
    } catch (err) {
        console.error('Error editing station:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.deleteStation = async (req, res) => {
    const id = req.params.id;
    try {
        const [result] = await db.query('DELETE FROM Station WHERE station_id = ?', [id]);
        if (result.affectedRows === 1) {
            res.json({ success: true });
        } else {
            res.status(400).json({ error: 'Failed to delete station' });
        }
    } catch (err) {
        console.error('Error deleting station:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};