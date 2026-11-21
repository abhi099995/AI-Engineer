# CI/CD Artifact Bucket - S3 Configuration
# This Terraform configuration creates a secure S3 bucket for storing CI/CD build artifacts.

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region where resources will be created"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name for tagging"
  type        = string
}

variable "owner" {
  description = "Owner name for tagging"
  type        = string
}

# S3 Bucket for CI/CD artifacts
resource "aws_s3_bucket" "ci_cd_artifacts" {
  bucket = "ci-cd-artifacts-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Environment = var.environment
    Owner       = var.owner
  }
}

# Block public access at the bucket level
resource "aws_s3_bucket_public_access_block" "ci_cd_artifacts_public_access" {
  bucket = aws_s3_bucket.ci_cd_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable versioning for the S3 bucket
resource "aws_s3_bucket_versioning" "ci_cd_artifacts_versioning" {
  bucket = aws_s3_bucket.ci_cd_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Enable server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "ci_cd_artifacts_encryption" {
  bucket = aws_s3_bucket.ci_cd_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Data source to get the current AWS account ID
data "aws_caller_identity" "current" {}