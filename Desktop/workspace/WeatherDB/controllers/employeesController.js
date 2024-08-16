const db = require('../config/connectDB');

exports.listEmployees = async (req, res) => {
    try {
        const [rows] = await db.query('SELECT * FROM Employee ORDER BY employee_name ASC');
        res.json({ employees: rows });
    } catch (err) {
        console.error('Error fetching employees:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.getEmployee = async (req, res) => {
    const id = req.params.id;
    try {
        const [rows] = await db.query('SELECT * FROM Employee WHERE id = ?', [id]);
        if (rows.length > 0) {
            res.json({ employee: rows[0] });
        } else {
            res.status(404).json({ error: 'Employee not found' });
        }
    } catch (err) {
        console.error('Error fetching employee:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.addEmployee = async (req, res) => {
    const { employee_name, department, position, contact } = req.body;
    try {
        const [result] = await db.query(
            'INSERT INTO Employee (employee_name, department, position, contact) VALUES (?, ?, ?, ?)',
            [employee_name, department, position, contact]
        );
        if (result.affectedRows === 1) {
            res.json({ success: true });
        } else {
            res.status(400).json({ error: 'Failed to add employee' });
        }
    } catch (err) {
        console.error('Error adding employee:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.editEmployee = async (req, res) => {
    const { id, employee_name, department, position, contact } = req.body;
    try {
        const [result] = await db.query(
            'UPDATE Employee SET employee_name = ?, department = ?, position = ?, contact = ? WHERE id = ?',
            [employee_name, department, position, contact, id]
        );
        if (result.affectedRows === 1) {
            res.json({ success: true });
        } else {
            res.status(400).json({ error: 'Failed to update employee' });
        }
    } catch (err) {
        console.error('Error editing employee:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

exports.deleteEmployee = async (req, res) => {
    const id = req.params.id;
    try {
        const [result] = await db.query('DELETE FROM Employee WHERE id = ?', [id]);
        if (result.affectedRows === 1) {
            res.json({ success: true });
        } else {
            res.status(400).json({ error: 'Failed to delete employee' });
        }
    } catch (err) {
        console.error('Error deleting employee:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
};