const predictionForm = document.getElementById("predictionForm");
const API_BASE_URL = "https://patient-health-outcome-predictor.onrender.com";

if (predictionForm) {

    predictionForm.addEventListener("submit", async function (event) {

        event.preventDefault();

        const resultText = document.getElementById("resultText");
        const probabilityText = document.getElementById("probabilityText");
        const probabilityBar = document.getElementById("probabilityBar");

        resultText.textContent = "Generating prediction...";
        probabilityText.textContent = "";

        probabilityBar.style.width = "0%";


        const patientData = {

            race: document.getElementById("race").value,

            gender: document.getElementById("gender").value,

            age: document.getElementById("age").value,

            admission_type_id: Number(
                document.getElementById("admission_type_id").value
            ),

            admission_source_id: Number(
                document.getElementById("admission_source_id").value
            ),

            time_in_hospital: Number(
                document.getElementById("time_in_hospital").value
            ),

            num_lab_procedures: Number(
                document.getElementById("num_lab_procedures").value
            ),

            num_procedures: Number(
                document.getElementById("num_procedures").value
            ),

            num_medications: Number(
                document.getElementById("num_medications").value
            ),

            number_outpatient: Number(
                document.getElementById("number_outpatient").value
            ),

            number_emergency: Number(
                document.getElementById("number_emergency").value
            ),

            number_inpatient: Number(
                document.getElementById("number_inpatient").value
            ),

            number_diagnoses: Number(
                document.getElementById("number_diagnoses").value
            ),

            A1Cresult: document.getElementById("A1Cresult").value,

            insulin: document.getElementById("insulin").value,

            change: document.getElementById("change").value,

            diabetesMed: document.getElementById("diabetesMed").value
        };


        try {

            const response = await fetch(
                `${API_BASE_URL}/predict`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify(patientData)
                }
            );


            if (!response.ok) {
                throw new Error("API request failed");
            }


            const result = await response.json();


            // Display prediction
            resultText.textContent = result.result;


            // Display probability
            probabilityText.textContent =
                "Readmission probability: " +
                result.probability +
                "%";


            // Update probability bar
            probabilityBar.style.width =
                result.probability + "%";


        } catch (error) {

            console.error("Prediction error:", error);

            resultText.textContent =
                "Unable to connect to the prediction server.";

            probabilityText.textContent =
                "Make sure the Flask API is running.";

            probabilityBar.style.width = "0%";
        }

    });

}