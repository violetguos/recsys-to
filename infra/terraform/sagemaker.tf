locals {
  data_s3_uri      = "s3://${aws_s3_bucket.data.id}/data"
  artifacts_s3_uri = "s3://${aws_s3_bucket.artifacts.id}/models"
}

resource "aws_ecr_repository" "training" {
  name                 = "${var.project_name}-${var.environment}-training"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  tags = { Environment = var.environment }
}

resource "aws_sagemaker_model_package_group" "this" {
  model_package_group_name        = "${var.project_name}-${var.environment}"
  model_package_group_description = "Baseline dummy model for Instacart basket prediction"
  tags                            = { Environment = var.environment }
}

resource "aws_sagemaker_notebook_instance" "dev" {
  name                = "${var.project_name}-${var.environment}-notebook"
  role_arn            = aws_iam_role.sagemaker_execution.arn
  instance_type       = "ml.t3.medium"
  platform_identifier = "notebook-al2-v1"
  tags                = { Environment = var.environment }
}

resource "aws_sagemaker_model" "dummy_baseline" {
  name                     = "${var.project_name}-${var.environment}-model"
  execution_role_arn       = aws_iam_role.sagemaker_execution.arn
  enable_network_isolation = true

  primary_container {
    image          = "${aws_ecr_repository.training.repository_url}:latest"
    model_data_url = "${local.artifacts_s3_uri}/training-output/output/model.tar.gz"
  }

  tags = { Environment = var.environment }
}

resource "aws_sagemaker_endpoint_configuration" "dummy_baseline" {
  name = "${var.project_name}-${var.environment}-ep-config"

  production_variants {
    variant_name           = "default"
    model_name             = aws_sagemaker_model.dummy_baseline.name
    initial_instance_count = 1
    instance_type          = "ml.t2.medium"
  }

  tags = { Environment = var.environment }
}

resource "aws_sagemaker_endpoint" "dummy_baseline" {
  name                 = "${var.project_name}-${var.environment}-ep"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.dummy_baseline.name
  tags                 = { Environment = var.environment }
}

resource "aws_sagemaker_model_package_group_policy" "this" {
  model_package_group_name = aws_sagemaker_model_package_group.this.model_package_group_name

  resource_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AddPermModelPackageGroup"
        Effect    = "Allow"
        Principal = { AWS = aws_iam_role.sagemaker_execution.arn }
        Action = [
          "sagemaker:DescribeModelPackage",
          "sagemaker:UpdateModelPackage",
          "sagemaker:CreateModelPackage",
        ]
        Resource = aws_sagemaker_model_package_group.this.arn
      }
    ]
  })
}

resource "null_resource" "trigger_training" {
  # This is an illustration — in production this would be a CI/CD pipeline
  # (CodePipeline / GitHub Actions) or a SageMaker Pipeline that calls
  # `aws sagemaker create-training-job` at deploy time.
  #
  # The equivalent AWS CLI invocation:
  #   aws sagemaker create-training-job \
  #     --training-job-name "${var.project_name}-${var.environment}-$(date +%s)" \
  #     --algorithm-specification TrainingImage=${aws_ecr_repository.training.repository_url}:latest,TrainingInputMode=File \
  #     --role-arn ${aws_iam_role.sagemaker_execution.arn} \
  #     --input-data-config ChannelName=training,DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=${local.data_s3_uri}}} \
  #     --output-data-config S3OutputPath=${local.artifacts_s3_uri}/training-output \
  #     --resource-config InstanceCount=1,InstanceType=${var.train_instance_type},VolumeSizeInGB=20 \
  #     --stopping-condition MaxRuntimeInSeconds=${var.train_max_run_seconds}

  triggers = {
    image_uri  = aws_ecr_repository.training.repository_url
    role_arn   = aws_iam_role.sagemaker_execution.arn
    data_uri   = local.data_s3_uri
    output_uri = local.artifacts_s3_uri
    instance   = var.train_instance_type
  }

  provisioner "local-exec" {
    command = <<EOT
      echo "To trigger training, run outside Terraform:"
      echo "aws sagemaker create-training-job ..."
      echo "See comment block above for the full command."
    EOT
  }
}
