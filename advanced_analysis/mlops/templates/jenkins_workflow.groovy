pipeline {
    agent {
        docker {
            image 'python:3.9'
        }
    }
    
    parameters {
        string(name: 'MODEL_PATH', defaultValue: 'models/model.pt', description: 'Path to the model file')
        string(name: 'FRAMEWORK', defaultValue: 'pytorch', description: 'Model framework (pytorch, tensorflow)')
        string(name: 'OPTIMIZATION_TECHNIQUES', defaultValue: 'quantization,pruning', description: 'Optimization techniques to apply (comma-separated)')
    }
    
    options {
        timeout(time: 2, unit: 'HOURS')
    }
    
    stages {
        stage('Install Dependencies') {
            steps {
                sh 'python -m pip install --upgrade pip'
                sh 'pip install -e .[all]'
            }
        }
        
        stage('Analyze Model') {
            steps {
                sh 'neural-scope analyze-model ${MODEL_PATH} --framework ${FRAMEWORK} --output model_analysis.json'
            }
        }
        
        stage('Optimize Model') {
            steps {
                sh 'neural-scope compress-model ${MODEL_PATH} --framework ${FRAMEWORK} --techniques ${OPTIMIZATION_TECHNIQUES} --output optimized_model.pt'
            }
        }
        
        stage('Validate Model') {
            steps {
                sh 'python tests/validate_model.py --model-path optimized_model.pt --dataset-path tests/data/test_data.csv'
                sh 'neural-scope analyze-model optimized_model.pt --framework ${FRAMEWORK} --output performance_report.json --analysis-types performance'
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'model_analysis.json,optimized_model.pt,performance_report.json', allowEmptyArchive: true
        }
        success {
            echo 'Model optimization completed successfully!'
        }
        failure {
            echo 'Model optimization failed!'
        }
    }
}
