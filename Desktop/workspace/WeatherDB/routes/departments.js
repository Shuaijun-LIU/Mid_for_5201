const express = require('express');
const router = express.Router();
const departmentsController = require('../controllers/departmentsController');
const employeesController = require('../controllers/employeesController');
const path = require('path');

router.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/departments.html'));
});

// 部门路由
router.get('/list', departmentsController.listDepartments);
router.get('/:id', departmentsController.getDepartment);
router.post('/add', departmentsController.addDepartment);
router.post('/edit', departmentsController.editDepartment);
router.delete('/delete/:id', departmentsController.deleteDepartment);

// 员工路由
router.get('/employees/list', employeesController.listEmployees);
router.get('/employees/:id', employeesController.getEmployee);
router.post('/employees/add', employeesController.addEmployee);
router.post('/employees/edit', employeesController.editEmployee);
router.delete('/employees/delete/:id', employeesController.deleteEmployee);

module.exports = router;