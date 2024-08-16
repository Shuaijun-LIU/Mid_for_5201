const db = require('../config/connectDB');

exports.listObservations = async (req, res) => {
    try {
        const [rows] = await db.query(`
            SELECT 
                Observation.*, 
                Time.Year, 
                Time.Month, 
                Time.Day, 
                Time.Hour 
            FROM 
                Observation 
            JOIN 
                Time 
            ON 
                Observation.time_id = Time.time_id 
            ORDER BY 
                Time.Year DESC, Time.Month DESC, Time.Day DESC, Time.Hour DESC
            LIMIT 10
        `);
        res.json({ observations: rows });
    } catch (err) {
        console.error('Error fetching observations:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.getObservation = async (req, res) => {
    const id = req.params.id;
    try {
        const [rows] = await db.query(`
            SELECT 
                Observation.*, 
                Time.Year, 
                Time.Month, 
                Time.Day, 
                Time.Hour 
            FROM 
                Observation 
            JOIN 
                Time 
            ON 
                Observation.time_id = Time.time_id 
            WHERE 
                Observation.time_id = ?
            LIMIT 10
        `, [id]);
        if (rows.length > 0) {
            res.json({ observation: rows[0] });
        } else {
            res.status(404).json({ error: 'Observation not found' });
        }
    } catch (err) {
        console.error('Error fetching observation:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.addObservation = async (req, res) => {
    const { time_id, station_id, observation_device_id, Temperature, Pressure } = req.body;
    try {
        const [result] = await db.query(
            'INSERT INTO Observation (time_id, station_id, observation_device_id, Temperature, Pressure) VALUES (?, ?, ?, ?, ?)',
            [time_id, station_id, observation_device_id, Temperature, Pressure]
        );
        if (result.affectedRows === 1) {
            res.json({ success: true });
        } else {
            res.status(400).json({ error: 'Failed to add observation' });
        }
    } catch (err) {
        console.error('Error adding observation:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.editObservation = async (req, res) => {
    const { time_id, station_id, observation_device_id, Temperature, Pressure } = req.body;
    try {
        const [result] = await db.query(
            'UPDATE Observation SET Temperature = ?, Pressure = ? WHERE time_id = ? AND station_id = ? AND observation_device_id = ?',
            [Temperature, Pressure, time_id, station_id, observation_device_id]
        );
        if (result.affectedRows === 1) {
            res.json({ success: true });
        } else {
            res.status(400).json({ error: 'Failed to update observation' });
        }
    } catch (err) {
        console.error('Error editing observation:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.deleteObservation = async (req, res) => {
    const { time_id, station_id, observation_device_id } = req.params;
    try {
        const [result] = await db.query(
            'DELETE FROM Observation WHERE time_id = ? AND station_id = ? AND observation_device_id = ?',
            [time_id, station_id, observation_device_id]
        );
        if (result.affectedRows === 1) {
            res.json({ success: true });
        } else {
            res.status(400).json({ error: 'Failed to delete observation' });
        }
    } catch (err) {
        console.error('Error deleting observation:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};