const axios = require('axios');
const Papa = require('papaparse');
const fs = require('fs');

// Base URL and data template
const url = "https://europe-west1-friendly-medley-412321.cloudfunctions.net/calculate-price";
const baseData = {
    "m": 50,
    "S0": 100,
    "K": 100,
    "H": 85,
    "q": 0,
    "confidence_level": 0.95,
    "n_paths": 3*10**7
};

// Parameters to iterate over
const sigmas = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6];
const Ts = Array.from({length: (5 - 0.2) / 0.1 + 1}, (_, i) => 0.2 + i * 0.1);
const rs = [0.1];

console.log(sigmas.length*Ts.length*rs.length)

async function makeApiCall(data) {
    try {
        const response = await axios.post(url, data);
        return {...response.data, ...data}; // Combine response with the input parameters
    } catch (error) {
        console.error('API call failed for:', data, error.message);
        return null; // Return null for failed requests to filter them out later
    }
}

async function fetchAllCombinationsAndSaveCSV() {
    console.time("Fetching and saving data");

    const promises = [];

    rs.forEach(r => {
        sigmas.forEach(sigma => {
            Ts.forEach(T => {
                const data = { ...baseData, r, sigma, T };
                promises.push(makeApiCall(data));
            });
        });
    });

    const results = await Promise.all(promises);
    const filteredResults = results.filter(result => result !== null); // Filter out failed requests

    // Convert the results to CSV
    const csv = Papa.unparse(filteredResults);

    // Save CSV to a file
    fs.writeFileSync('results_parallel.csv', csv);
    console.log('CSV file with all combinations has been saved.');

    console.timeEnd("Fetching and saving data");
}

fetchAllCombinationsAndSaveCSV();
