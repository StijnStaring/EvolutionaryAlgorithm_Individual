import pylab as plt

def figure_layout(figsize=(10, 8), titel="", xlabel="", ylabel="", fontsize_titel=18, fontsize_axis=16,fontsize_legend=14, fontsize_ticks=16):
	plt.figure(figsize=figsize)
	ax1 = plt.gca()
	plt.rc('legend', fontsize=fontsize_legend)
	plt.title(titel, fontsize=fontsize_titel, fontweight='bold')
	plt.grid(True)
	plt.xlabel(xlabel, fontsize=fontsize_axis)
	plt.ylabel(ylabel, fontsize=fontsize_axis)
	for tick in ax1.xaxis.get_major_ticks():
		tick.label1.set_fontsize(fontsize_ticks)
	#         tick.label1.set_fontweight('bold')
	for tick in ax1.yaxis.get_major_ticks():
		tick.label1.set_fontsize(fontsize_ticks)
	#     tick.label1.set_fontweight('bold')

	return ax1