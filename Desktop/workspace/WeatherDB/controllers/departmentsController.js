const db = require('../config/connectDB');

exports.listDepartments = async (req, res) => {
    try {
        const [rows] = await db.query('SELECT * FROM Department ORDER BY department_name ASC');
        res.json({ departments: rows });
    } catch (err) {
        console.error('Error fetching departments:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.getDepartment = async (req, res) => {
    const id = req.params.id;
    try {
        const [rows] = await db.query('SELECT * FROM Department WHERE id = ?', [id]);
        if (rows.length > 0) {
            res.json({ department: rows[0] });
        } else {
            res.status(404).json({ error: 'Department not found' });
        }
    } catch (err) {
        console.error('Error fetching department:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.addDepartment = async (req, res) => {
    const { department_name, head, creation_date, description } = req.body;
    try {
        const [result] = await db.query(
            'INSERT INTO Department (department_name, head, creation_date, description) VALUES (?, ?, ?, ?)',
            [department_name, head, creation_date, description]
        );
        if (result.affectedRows === 1) {
            res.json({ success: true });
        } else {
            res.status(400).json({ error: 'Failed to add department' });
        }
    } catch (err) {
        console.error('Error adding department:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.editDepartment = async (req, res) => {
    const { id, department_name, head, creation_date, description } = req.body;
    try {
        const [result] = await db.query(
            'UPDATE Department SET department_name = ?, head = ?, creation_date = ?, description = ? WHERE id = ?',
            [department_name, head, creation_date, description, id]
        );
        if (result.affectedRows === 1) {
            res.json({ success: true });
        } else {
            res.status(400).json({ error: 'Failed to update department' });
        }
    } catch (err) {
        console.error('Error editing department:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.deleteDepartment = async (req, res) => {
    const id = req.params.id;
    try {
        const [result] = await db.query('DELETE FROM Department WHERE id = ?', [id]);
        if (result.affectedRows === 1) {
            res.json({ success: true });
        } else {
            res.status(400).json({ error: 'Failed to delete department' });
        }
    } catch (err) {
        console.error('Error deleting department:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};