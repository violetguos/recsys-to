resource "aws_s3_bucket" "data" {
  bucket        = "${var.project_name}-${var.environment}-data"
  force_destroy = true
  tags          = { Name = "${var.project_name}-data", Environment = var.environment }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket" "artifacts" {
  bucket        = "${var.project_name}-${var.environment}-artifacts"
  force_destroy = true
  tags          = { Name = "${var.project_name}-artifacts", Environment = var.environment }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_object" "csv_data" {
  for_each = {
    aisles      = "data/aisles.csv"
    departments = "data/departments.csv"
    products    = "data/products.csv"
    orders      = "data/orders.csv"
    order_prior = "data/order_products__prior.csv"
    order_train = "data/order_products__train.csv"
  }
  bucket = aws_s3_bucket.data.id
  key    = each.value
  source = "../../data/${basename(each.value)}"
  etag   = filemd5("../../data/${basename(each.value)}")
}
