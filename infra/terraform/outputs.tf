output "data_bucket_arn" {
  description = "S3 bucket holding raw CSV datasets"
  value       = aws_s3_bucket.data.arn
}

output "artifacts_bucket_arn" {
  description = "S3 bucket storing trained model artifacts"
  value       = aws_s3_bucket.artifacts.arn
}

output "sagemaker_execution_role_arn" {
  description = "IAM role assumed by SageMaker jobs"
  value       = aws_iam_role.sagemaker_execution.arn
}

output "model_package_group_name" {
  description = "SageMaker Model Package Group for version tracking"
  value       = aws_sagemaker_model_package_group.this.model_package_group_name
}

output "ecr_repository_url" {
  description = "ECR repository for the training container image"
  value       = aws_ecr_repository.training.repository_url
}

output "endpoint_name" {
  description = "SageMaker endpoint name for model inference"
  value       = aws_sagemaker_endpoint.dummy_baseline.name
}

output "model_name" {
  description = "SageMaker Model resource name"
  value       = aws_sagemaker_model.dummy_baseline.name
}

output "notebook_instance_name" {
  description = "SageMaker notebook instance for ad-hoc exploration"
  value       = aws_sagemaker_notebook_instance.dev.name
}
