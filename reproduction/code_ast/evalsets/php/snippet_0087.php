<div class="row">
	<div class="col-md-3">
		<div class="box box-primary">
			<div class="box-header">
				<h3 class="box-title">Details</h3>
			</div>
			<div class="box-body box-profile">
				<ul class="list-group list-group-unbordered">
					<li class="list-group-item">
						<b>Year Effective</b> <a class="pull-right"><?php echo $phic_matrix['effectivity_date']; ?></a>
					</li>
					<li class="list-group-item">
						<b>Description</b> <a class="pull-right"><?php echo $phic_matrix['description']; ?></a>
					</li>
					<li class="list-group-item">
						<b>Status Label</b> <a class="pull-right"><?php echo $phic_matrix['status_label']; ?></a>
					</li>
					<li class="list-group-item">
						<b>Created</b> <a class="pull-right"><?php echo date('F d, Y', strtotime($phic_matrix['created'])); ?></a>
					</li>
				</ul>
			</div>
		</div>
	</div>
	<div class="col-md-9">
		<div class="box box-primary">
			<div class="box-header">
				<h3 class="box-title">PHIC Rates</h3>
				<div class="box-tools">
					<a href="<?php echo site_url('phic_contribution_rates/load_form/' . $phic_matrix['id']); ?>" class="btn btn-box-tool" data-toggle="modal" data-target="#modal-add-phic">
						<i class="fa fa-plus"></i> <span class="text-blue">Add New PHIC Rate</span>
					</a>
				</div>
			</div>
			<div class="box-body">
				<div class="table-responsive">
					<table class="table table-bordered" id="datatables-tax-rates">
						<thead>
							<tr>
								<th>Action</th>
								<th>Minimum Monthly Salary</th>
								<th>Maximum Monthly Salary</th>
								<th>Employee Shares</th>
								<th>Employer Shares</th>
								<th>Total Monthly Premium</th>
							</tr>
						</thead>
						<tbody>
							<?php if ( ! empty($phic_rates)): ?>
							<?php foreach ($phic_rates as $index => $phic_rate): ?>
							<tr>
								<td>
									<a href="<?php echo site_url('phic_contribution_rates/details/' . $phic_rate['phic_rate_id']); ?>" class="btn btn-link btn-sm" data-toggle="modal" data-target="#modal-details-<?php echo $index; ?>">
										<i class="fa fa-eye"></i> View
									</a>
									<a href="<?php echo site_url('phic_contribution_rates/confirmation/edit/' . $phic_rate['phic_rate_id']); ?>" class="btn btn-link btn-sm" data-toggle="modal" data-target="#modal-confirmation-<?php echo $index; ?>">
										<i class="fa fa-edit"></i> Edit
									</a>
									<a href="<?php echo site_url('phic_contribution_rates/confirmation/' . $phic_rate['pr_status_url'] . '/' . $phic_rate['phic_rate_id']); ?>" class="btn btn-link btn-sm" data-toggle="modal" data-target="#modal-update-status-<?php echo $index; ?>">
										<i class="fa <?php echo $phic_rate['pr_status_icon']; ?>"></i> <?php echo $phic_rate['pr_status_action']; ?>
									</a>
								</td>
								<td><?php echo $phic_rate['pr_minimum_range']; ?></td>
								<td><?php echo $phic_rate['pr_maximum_range']; ?></td>
								<td><?php echo $phic_rate['pr_employee_share']; ?></td>
								<td><?php echo $phic_rate['pr_employer_share']; ?></td>
								<td><?php echo $phic_rate['pr_total_monthly_premium']; ?></td>
							</tr>
							
							<div class="modal fade" id="modal-details-<?php echo $index; ?>">
								<div class="modal-dialog">
									<div class="modal-content"></div>
								</div>
							</div>

							<div class="modal fade" id="modal-confirmation-<?php echo $index; ?>">
								<div class="modal-dialog">
									<div class="modal-content"></div>
								</div>
							</div>

							<div class="modal fade" id="modal-update-status-<?php echo $index; ?>">
								<div class="modal-dialog">
									<div class="modal-content"></div>
								</div>
							</div>

							<?php endforeach ?>
							<?php endif ?>
						</tbody>
					</table>
				</div>
			</div>
			<div class="modal fade" id="modal-add-phic">
				<div class="modal-dialog">
					<div class="modal-content"></div>
				</div>
			</div>
			<?php if ($show_modal): ?>
				<div class="modal fade" id="modal-edit-tax-rate">
					<div class="modal-dialog">
						<div class="modal-content"><?php $this->load->view($modal_file_path); ?></div>
					</div>
				</div>
				<script type="text/javascript">
					$(function() {
						$('#modal-edit-tax-rate').modal({
							backdrop: false,
							keyboard: false
						});
					});
				</script>
			<?php endif ?>
		</div>
	</div>


</div>