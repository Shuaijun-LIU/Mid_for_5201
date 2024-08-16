// scripts.js
document.addEventListener('DOMContentLoaded', function() {
    // Fetch and display the dashboard data
    fetch('/api/dashboard')
        .then(response => response.json())
        .then(data => {
            document.getElementById('observation-count').textContent = data.observationCount;
            document.getElementById('station-count').textContent = data.stationCount;
            document.getElementById('device-status').textContent = data.deviceStatus;
        });

    // Add event listener for executing custom SQL queries
    document.getElementById('execute-query').addEventListener('click', function() {
        const query = document.getElementById('sql-query').value;
        executeQuery(query);
    });

    // Add event listeners for quick queries
    const quickQueryButtons = document.querySelectorAll('.quick-query');
    quickQueryButtons.forEach(button => {
        button.addEventListener('click', function() {
            const query = button.textContent;
            document.getElementById('sql-query').value = query;
            executeQuery(query);
        });
    });

    function executeQuery(query) {
        fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('query-result').textContent = 'Error: ' + data.error;
            } else {
                document.getElementById('query-result').textContent = JSON.stringify(data.result, null, 2);
            }
        });
    }
});