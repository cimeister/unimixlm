using System;
using System.ComponentModel;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using FFImageLoading.Extensions;
using FFImageLoading.Forms.Args;
using FFImageLoading.Helpers;
using FFImageLoading.Work;
using Xamarin.Forms;
using Xamarin.Forms.Internals;
using Xamarin.Forms.Platform.WPF;
using Image = System.Windows.Controls.Image;

namespace FFImageLoading.Forms.Platform
{
	[Preserve(AllMembers = true)]
	public class CachedImageRenderer : ViewRenderer<CachedImage, Image>
	{
		[RenderWith(typeof(CachedImageRenderer))]
		internal class _CachedImageRenderer
		{
		}

		private bool _measured;

		private IScheduledWork _currentTask;

		private ImageSourceBinding _lastImageSource;

		private bool _isDisposed = false;

		public static void Init()
		{
			CachedImage.IsRendererInitialized = true;
			ScaleHelper.InitAsync();
		}

		public override SizeRequest GetDesiredSize(double widthConstraint, double heightConstraint)
		{
			BitmapSource bitmapSource = Control.Source as BitmapSource;
			if (bitmapSource == null)
			{
				return default(SizeRequest);
			}
			_measured = true;
			return new SizeRequest(new Xamarin.Forms.Size(bitmapSource.PixelWidth, bitmapSource.PixelHeight));
		}

		protected override void OnElementChanged(ElementChangedEventArgs<CachedImage> e)
		{
			base.OnElementChanged(e);
			if (Control == null && Element != null && !_isDisposed)
			{
				Image nativeControl = new Image
				{
					Stretch = GetStretch((Aspect)1)
				};
				SetNativeControl(nativeControl);
				Control.HorizontalAlignment = HorizontalAlignment.Center;
				Control.VerticalAlignment = VerticalAlignment.Center;
			}
			if (e.OldElement != null)
			{
				e.OldElement.InternalReloadImage = null;
				e.OldElement.InternalCancel = null;
				e.OldElement.InternalGetImageAsJPG = null;
				e.OldElement.InternalGetImageAsPNG = null;
			}
			if (e.NewElement != null)
			{
				e.NewElement.InternalReloadImage = ReloadImage;
				e.NewElement.InternalCancel = CancelIfNeeded;
				e.NewElement.InternalGetImageAsJPG = GetImageAsJpgAsync;
				e.NewElement.InternalGetImageAsPNG = GetImageAsPngAsync;
				UpdateAspect();
				UpdateImage(Control, Element, e.OldElement);
			}
		}

		protected override void Dispose(bool disposing)
		{
			if (!_isDisposed)
			{
				_isDisposed = true;
				if (Control != null)
				{

				}
				CancelIfNeeded();
			}
			base.Dispose(disposing);
		}

		protected override void OnElementPropertyChanged(object sender, PropertyChangedEventArgs e)
		{
			base.OnElementPropertyChanged(sender, e);
			if (e.PropertyName == CachedImage.SourceProperty.PropertyName)
			{
				UpdateImage(Control, Element, null);
			}
			else if (e.PropertyName == CachedImage.AspectProperty.PropertyName)
			{
				UpdateAspect();
			}
		}

		private void OnImageOpened(object sender, RoutedEventArgs routedEventArgs)
		{
			if (_measured)
			{
				if (Element != null)
				{
					((IVisualElementController)Element).InvalidateMeasure((InvalidationTrigger)16);
				}
			}
		}

		private async void UpdateImage(Image imageView, CachedImage image, CachedImage previousImage)
		{
			CancelIfNeeded();
			if (image == null || imageView == null || _isDisposed)
			{
				return;
			}
			ImageSourceBinding ffSource = await ImageSourceBinding.GetImageSourceBinding(image.Source, image)
												.ConfigureAwait(continueOnCapturedContext: false);
			if (ffSource == null)
			{
				if (_lastImageSource != null)
				{
					_lastImageSource = null;
					imageView.Source = null;
				}
				return;
			}
			if (previousImage != null && !ffSource.Equals(_lastImageSource))
			{
				_lastImageSource = null;
				imageView.Source = null;
			}
			image.SetIsLoading(isLoading: true);
			ImageSourceBinding placeholderSource = await ImageSourceBinding.GetImageSourceBinding(image.LoadingPlaceholder, image).ConfigureAwait(continueOnCapturedContext: false);
			ImageSourceBinding errorPlaceholderSource = await ImageSourceBinding.GetImageSourceBinding(image.ErrorPlaceholder, image).ConfigureAwait(continueOnCapturedContext: false);
			image.SetupOnBeforeImageLoading(out var imageLoader, ffSource, placeholderSource, errorPlaceholderSource);
			if (imageLoader != null)
			{
				Action<IScheduledWork> finishAction = imageLoader.OnFinish;
				Action<ImageInformation, LoadingResult> sucessAction = imageLoader.OnSuccess;
				imageLoader.Finish(delegate(IScheduledWork work)
				{
					finishAction?.Invoke(work);
					ImageLoadingSizeChanged(image, isLoading: false);
				});
				imageLoader.Success(delegate(ImageInformation imageInformation, LoadingResult loadingResult)
				{
					sucessAction?.Invoke(imageInformation, loadingResult);
					_lastImageSource = ffSource;
				});
				imageLoader.LoadingPlaceholderSet(delegate
				{
					ImageLoadingSizeChanged(image, isLoading: true);
				});
				if (!_isDisposed)
				{
					_currentTask = imageLoader.Into(imageView);
				}
			}
		}

		private void UpdateAspect()
		{
			if (Control != null && Element != null && !_isDisposed)
			{
				Control.Stretch = GetStretch(Element.Aspect);
			}
		}

		private static Stretch GetStretch(Aspect aspect)
		{
			if ((int)aspect != 1)
			{
				if ((int)aspect == 2)
				{
					return Stretch.Fill;
				}
				return Stretch.Uniform;
			}
			return Stretch.UniformToFill;
		}

		private async void ImageLoadingSizeChanged(CachedImage element, bool isLoading)
		{
			if (element == null || _isDisposed)
			{
				return;
			}
			await ImageService.Instance.Config.MainThreadDispatcher.PostAsync(delegate
			{
				if (element != null && !_isDisposed)
				{
					((IVisualElementController)element).InvalidateMeasure((InvalidationTrigger)16);
					if (!isLoading)
					{
						element.SetIsLoading(isLoading);
					}
				}
			}).ConfigureAwait(continueOnCapturedContext: false);
		}

		private void ReloadImage()
		{
			UpdateImage(Control, Element, null);
		}

		private void CancelIfNeeded()
		{
			try
			{
				_currentTask?.Cancel();
			}
			catch (Exception)
			{
			}
		}

		private async Task<byte[]> GetImageAsJpgAsync(GetImageAsJpgArgs args)
		{
			System.Windows.Media.ImageSource source = Control.Source;
			WriteableBitmap bitmap = source as WriteableBitmap;
			if (bitmap != null)
			{
				return await bitmap.AsJpegBytesAsync(args.Quality);
			}
			return null;
		}

		private async Task<byte[]> GetImageAsPngAsync(GetImageAsPngArgs args)
		{
			System.Windows.Media.ImageSource source = Control.Source;
			WriteableBitmap bitmap = source as WriteableBitmap;
			if (bitmap != null)
			{
				return await bitmap.AsPngBytesAsync();
			}
			return null;
		}
	}
}