const db = require('../config/connectDB');

exports.getDashboardData = async (req, res) => {
    try {
        const [observationRows] = await db.query('SELECT COUNT(*) AS count FROM Observation');
        const [stationRows] = await db.query('SELECT COUNT(*) AS count FROM Station');
        const [deviceRows] = await db.query('SELECT COUNT(*) AS count FROM ObservationDevice WHERE status = "Online"');
        const [employeeRows] = await db.query('SELECT COUNT(*) AS count FROM SpecialEvent');
        
    
        const [latestObservation] = await db.query('SELECT MAX(time_id) AS latest_time_id FROM Observation');

        
        const [timeRows] = await db.query('SELECT * FROM Time WHERE time_id = ?', [latestObservation[0].latest_time_id]);

        
        const latestObservationDate = timeRows.length > 0 
            ? `${timeRows[0].Year}-${timeRows[0].Month}-${timeRows[0].Day} ${timeRows[0].Hour}:00`
            : 'No data available';

        res.json({
            observationCount: observationRows[0].count,
            stationCount: stationRows[0].count,
            deviceStatus: deviceRows[0].count,
        });
    } catch (err) {
        console.error('Error fetching dashboard data:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.executeQuery = async (req, res) => {
    const query = req.body.query;
    try {
        const [rows] = await db.query(query);
        res.json({ result: rows });
    } catch (err) {
        console.error('Error executing query:', err);
        res.status(400).json({ error: 'Invalid SQL query' });
    }
};